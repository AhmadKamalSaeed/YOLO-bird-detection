# YOLO Bird Detection & Tracking

Detects and tracks birds in video footage using YOLOv8x and BoT-SORT. Outputs a per-frame CSV of bounding boxes and track IDs, and optionally an annotated video with overlays.

---

## Requirements

- Python 3.13
- Apple Silicon (MPS) recommended. NVIDIA GPU (CUDA) or CPU also work — set `DEVICE` in `main.py`.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

YOLOv8x weights (`yolov8x.pt`, ~137 MB) are downloaded automatically on first run.

---

## Usage

Edit the two configuration lines at the top of `run.sh`:

```bash
VIDEO="/path/to/your/video.mp4"
OUTPUT_DIR="/path/to/output/folder"
```

Then run:

```bash
./run.sh
```

### Options

| Variable in `run.sh` | Default | Effect |
|---|---|---|
| `SHOW_PREVIEW` | `false` | Open a live cv2 window while processing |
| `SAVE_VIDEO` | `true` | Write an annotated `.mp4` alongside the CSV |

Set `SAVE_VIDEO=false` for a faster CSV-only run (no video written, frame stride kicks back in).

### Output files

Each run generates a unique 8-character ID shared between both files so they stay paired:

```
output_<run_id>.csv          — per-frame detections
output_video_<run_id>.mp4    — annotated video (if SAVE_VIDEO=true)
```

### CSV schema

| Column | Type | Description |
|---|---|---|
| `frame_id` | int | Zero-indexed frame number |
| `bbox` | string | `[cx, cy, w, h]` in pixels, original resolution |
| `track_id` | int | Persistent identity assigned by BoT-SORT |
| `class` | int | COCO class ID (0 = person, 2 = car, 14 = bird, 16 = dog/cat) |
| `timestamp` | float | `frame_id / fps`, seconds |

---

## Validation

`validate.py` runs automatically at the end of every `main.py` run. It can also be run standalone against any existing CSV:

```bash
python validate.py output_<run_id>.csv --video /path/to/video.mp4
```

### Checks

| Category | Check |
|---|---|
| Structure | Required columns present; no nulls in key columns |
| IDs | `frame_id ≥ 0`, `track_id ≥ 1`; no duplicate `(frame_id, track_id)` pairs |
| Timestamps | Non-decreasing across frames; consistent with `frame_id / fps` |
| Bounding boxes | Parseable; positive width and height; coordinates within frame bounds |

Exit code `0` = all checks passed. Non-zero = at least one check failed.

---

## Tuning

All detection knobs are constants at the top of `main.py`:

| Constant | Default | Effect |
|---|---|---|
| `DEVICE` | `'mps'` | Inference device: `'mps'`, `'cuda'`, `'cpu'` |
| `IMGSZ` | `2560` | Inference resolution. Lower = faster, worse recall on small birds |
| `FRAME_STRIDE` | `2` | Process every Nth frame. Ignored (forced to 1) when `SAVE_VIDEO=true` |
| `CONF` | `0.10` | Min detection confidence. Lower = more weak detections kept |
| `IOU` | `0.4` | NMS threshold. Lower = packed birds less likely to suppress each other |
| `AUGMENT` | `True` | Test-time augmentation (~3 passes per frame, +recall, ~3× slower) |

**Fastest run** (CSV only, no augmentation, half the frames):

```python
IMGSZ = 1920
AUGMENT = False
```

```bash
SAVE_VIDEO=false ./run.sh   # FRAME_STRIDE=2 kicks in automatically
```

---

## Files

```
main.py          — detection, tracking, CSV and video output
validate.py      — output validation (standalone or auto-run)
run.sh           — entry point with configurable flags
requirements.txt — Python dependencies
WRITEUP.md       — technical writeup: approach, tradeoffs, edge deployment, AI usage
```
