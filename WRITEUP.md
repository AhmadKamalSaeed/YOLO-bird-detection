# Bird Detection & Tracking — Technical Writeup

## What the code does

`main.py` reads a video file, runs YOLOv8x frame-by-frame to detect objects, assigns each detection a persistent track ID using BoT-SORT, and writes two output files per run:

- `output_<run_id>.csv` — one row per detection per frame: `frame_id`, `bbox` (cx, cy, w, h), `track_id`, `class`, `timestamp`.
- `output_video_<run_id>.mp4` — the original footage with bounding boxes, track IDs, and class labels overlaid.

`validate.py` runs automatically at the end of each `main.py` run and also works standalone. It checks 9 properties of the CSV: schema completeness, no missing values in key columns, valid ID ranges, no duplicate track assignments within a frame, monotonic timestamps, timestamp/frame-id consistency, bbox parseability, positive dimensions, and coordinates within the actual frame bounds.

---

## Thought process and iteration

**Starting point:** the initial script used `yolov8n` (nano, the smallest model) at `imgsz=640` (half the video resolution) with default settings. It produced ~8 birds per frame on average; visual inspection showed most of the flock — particularly the small and distant birds — had no boxes at all.

**Problem diagnosis:** the root cause was resolution collapse. The source video is 1920×1080. At `imgsz=640`, each frame is downscaled ~3× before the model sees it. A pigeon that's 30px tall in the original becomes ~10px — below the reliable detection floor of any COCO-pretrained model. The tracker was then inheriting every miss: a bird the detector never saw couldn't get a track ID, and when it was occasionally detected it was assigned a new ID rather than continuing its existing one. The `max_track_id / unique_track_ids` ratio was 8.6×, meaning each real bird was being fragmented into ~8 short-lived identities.

**Iteration:**

1. Swapped `yolov8n` → `yolov8x`. Larger model, better recall on small objects. Avg birds/frame: 8 → 22.
2. Raised `imgsz` to 1920 (native video resolution). Birds that were 10px blobs now appear at 30px — enough for the model to reliably fire. Avg birds/frame: 22 → ~35.
3. Lowered `conf` from 0.25 → 0.10. Low-confidence detections on small distant birds are worth keeping; a weak true positive is better than a miss when the goal is census-style counting.
4. Lowered NMS `iou` from 0.7 → 0.4. Pigeons standing in a flock overlap each other's bounding boxes by 40–60%. At the default threshold NMS was suppressing real birds as if they were duplicate detections of the same bird. Lowering it lets adjacent birds survive NMS independently.
5. Switched tracker from ByteTrack → BoT-SORT. BoT-SORT uses a re-identification module and holds identity across occlusion frames better, which directly addresses the ID fragmentation problem.
6. Added `augment=True`. Test-time augmentation runs the model at multiple scales and with horizontal flipping, merging the results. +10–20% recall at the cost of ~3× slower inference.

---

## Tradeoffs made

**Model size vs. speed:** `yolov8x` is the best COCO model for recall but is ~20× larger and ~4× slower than `yolov8n`. On Apple Silicon with MPS the full pipeline at `imgsz=2560` with augmentation takes roughly 1–2 seconds per frame — acceptable for an offline batch job processing a 16-second clip, but not real-time.

**Inference resolution vs. speed:** `imgsz=2560` is above native video resolution. This forces the model to work on an upscaled image, which helps with very small far-away birds but adds processing time quadratically. Dropping to `imgsz=1920` (native) or `imgsz=1280` would significantly speed things up with a modest recall cost.

**Confidence threshold vs. false positives:** `conf=0.10` will let some non-bird detections through. The CSV reflects this — class 2 (car), class 0 (person), and others appear alongside class 14 (bird). For the purposes of this task the priority was not missing birds; a consumer of the CSV can filter by `class == 14` if bird-only data is needed.

**`SAVE_VIDEO` forces stride=1:** when saving an annotated video, every frame must be processed so the output plays at the correct speed. This doubles processing time compared to `FRAME_STRIDE=2`. When you only need the CSV (not the video), setting `SAVE_VIDEO=False` halves run time at no cost to data quality.

**No SAHI:** Sliced Aided Hyper Inference — cutting each frame into overlapping tiles and running detection on each independently — would give the largest recall boost on the dense back-of-flock birds. It was evaluated but not implemented because it requires feeding detections into a standalone tracker (breaking `model.track()`'s built-in state), adding significant code complexity and per-frame latency. It remains the correct next step if the current recall level is insufficient.

---

## How this would run on edge hardware near the cameras

The core question for edge deployment is the latency budget: does the processing need to be real-time (every frame, live) or near-real-time (small lag acceptable)?

**For near-real-time batch processing (most realistic for wildlife monitoring):**

- The pipeline as written is already batch-friendly. A Raspberry Pi 5 or NVIDIA Jetson Orin running `yolov8s` or `yolov8m` at `imgsz=640` would achieve 5–15 fps — sufficient if the camera records at 24fps and the edge device processes in parallel, or processes the overnight footage during daylight hours.
- `FRAME_STRIDE=3` or `4` would keep up with live input on constrained hardware. The tracker tolerates gaps because BoT-SORT's `track_buffer` holds identities across missed frames.
- The model weights (`yolov8x.pt` at ~137 MB) fit comfortably in the RAM of any Jetson device. For a severely constrained device (Pi 4, 4GB), swap to `yolov8n.pt` (6 MB) and accept the recall trade-off.

**For real-time deployment:**

- Export the model to TensorRT (NVIDIA) or CoreML (Apple) for hardware-specific inference. Ultralytics supports both via `model.export(format='engine')` or `model.export(format='coreml')`. This typically gives 3–10× latency improvement over PyTorch.
- Run detection on a GPU-enabled Jetson (AGX Orin, NX) and stream results over MQTT or a local socket to a lightweight aggregation service.
- The CSV output would be replaced by a streaming write to a local SQLite database or a time-series store, with the same schema.

**Practical constraints:**
- `imgsz=2560` with augmentation is not feasible on embedded hardware. `imgsz=640`, no augmentation, `yolov8n` or `yolov8s` is the realistic edge configuration.
- Power consumption matters. Jetson Orin draws ~15–60W under load. A camera system running continuously needs this budgeted.
- Network-connected edge devices should be able to push only the CSV rows (a few KB/frame) rather than raw video, keeping bandwidth requirements minimal.

---

## How AI tools were used

Claude (via Cursor) was used throughout this task as a pair programmer and reasoning partner. Specifically:

- **Diagnosis:** after the first run showed low detection counts, Claude analysed the output CSV statistics (avg birds/frame, max_track_id vs unique_track_ids ratio) and identified inference resolution as the root cause rather than model capacity — a non-obvious diagnosis.
- **Parameter tuning:** Claude explained the mechanical effect of each parameter (`conf`, `iou`, `imgsz`, `augment`, tracker choice) and the tradeoff of each change, letting me make informed decisions rather than random adjustments.
- **Code structure:** the `SAVE_VIDEO`/`SHOW_PREVIEW`/`FRAME_STRIDE` flag system, the `effective_stride` override logic, the `VideoWriter` initialisation with an `isOpened()` guard, and the `run_validation()` importable function were all designed with Claude's input.
- **Validation script:** Claude drafted the 9 checks in `validate.py`, with reasoning for each — including the `timestamp/frame_id` ratio consistency check and the 1px tolerance on the bbox bounds check.

The AI did not run the code or make unilateral decisions. Every parameter change was reviewed, and several suggestions were explicitly rejected or modified (e.g., SAHI was discussed but intentionally deferred as too complex for this scope). Claude's role was to surface options, quantify tradeoffs, and implement decisions once made — not to determine the direction.
