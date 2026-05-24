import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# Tuning knobs — adjust to trade speed for recall.
DEVICE = 'mps'              # 'mps' (Apple Silicon GPU), 'cuda' (NVIDIA), 'cpu' (fallback)
IMGSZ = 2560                # upscaled inference; small distant birds get more pixels to bite on
FRAME_STRIDE = 2            # process every Nth frame; ignored (forced to 1) when SAVE_VIDEO=True
CONF = 0.10                 # min detection confidence; lower = more weak detections kept
IOU = 0.4                   # NMS IoU; lower = packed birds less likely to suppress each other
AUGMENT = True              # test-time augmentation (multi-scale + flip); ~2-3x slower, +recall
SHOW_PREVIEW = False        # set True to watch the cv2.imshow window while it runs
SAVE_VIDEO = True           # write annotated video with overlays; forces every frame to be processed


def load_local_env(env_path: Path = Path('local.env')) -> None:
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO bird detection and tracking')
    parser.add_argument(
        '--video',
        default=os.environ.get('VIDEO_PATH'),
        help='Path to input video (or set VIDEO_PATH env var)',
    )
    parser.add_argument(
        '--output-dir',
        default=os.environ.get('OUTPUT_DIR', '.'),
        help='Directory for output video and CSV (default: current dir)',
    )
    args = parser.parse_args()
    if not args.video:
        parser.error('Provide --video or set VIDEO_PATH')
    return args


load_local_env()
args = parse_args()
_run_id = uuid.uuid4().hex[:8]
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_PATH = str(output_dir / f'output_video_{_run_id}.mp4')
OUTPUT_CSV_PATH = str(output_dir / f'output_{_run_id}.csv')

# Load the model
model = YOLO('yolov8x.pt')

cap = cv2.VideoCapture(args.video)

# Get frames per second (FPS) to calculate the timestamp later
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

effective_stride = 1 if SAVE_VIDEO else FRAME_STRIDE
if SAVE_VIDEO and FRAME_STRIDE != 1:
    print(f'SAVE_VIDEO is on: processing every frame (FRAME_STRIDE={FRAME_STRIDE} ignored).')

total_to_process = max(1, (total_frames + effective_stride - 1) // effective_stride)

writer = None
if SAVE_VIDEO:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter for {OUTPUT_VIDEO_PATH}')

# This list will hold all our data rows before saving to CSV
results_data = []
frame_id = 0
processed = 0
start_time = time.perf_counter()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print('Video finished processing.')
        break

    # Skip frames per effective_stride without losing the true frame_id for timestamps.
    if frame_id % effective_stride != 0:
        frame_id += 1
        continue

    # persist=True keeps tracker IDs across frames; botsort holds identity better
    # than bytetrack when detections flicker on small/occluded birds.
    results = model.track(
        frame,
        persist=True,
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=1000,
        augment=AUGMENT,
        tracker='botsort.yaml',
        verbose=False,
    )
    
    # Extract data ONLY if objects with tracking IDs are found in the frame
    if results[0].boxes.id is not None: 
        boxes = results[0].boxes.xywh.cpu().numpy() # Bounding boxes in (x, y, w, h) format
        track_ids = results[0].boxes.id.cpu().numpy() # Persistent Track IDs
        classes = results[0].boxes.cls.cpu().numpy() # Class labels (e.g., 14 for bird)
        
        # Loop through every object detected in this specific frame
        for box, track_id, cls in zip(boxes, track_ids, classes):
            # Format the bounding box clearly
            bbox_str = f"[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]"
            timestamp = frame_id / fps
            
            # Save the exact fields Spoor requested
            results_data.append({
                "frame_id": frame_id,
                "bbox": bbox_str,
                "track_id": int(track_id),
                "class": int(cls),
                "timestamp": round(timestamp, 3)
            })

    if SAVE_VIDEO or SHOW_PREVIEW:
        annotated_frame = results[0].plot()

    if SAVE_VIDEO:
        writer.write(annotated_frame)

    if SHOW_PREVIEW:
        cv2.imshow('YOLO Tracking', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        processed += 1
        elapsed = time.perf_counter() - start_time
        rate = processed / elapsed if elapsed > 0 else 0.0
        eta = (total_to_process - processed) / rate if rate > 0 else 0.0
        det_count = 0 if results[0].boxes.id is None else len(results[0].boxes.id)
        sys.stdout.write(
            f'\r[{processed}/{total_to_process}] '
            f'{processed / total_to_process * 100:5.1f}%  '
            f'frame={frame_id}  dets={det_count:3d}  '
            f'{rate:4.2f} fps  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s'
        )
        sys.stdout.flush()

    frame_id += 1

# Clean up windows and video writer
cap.release()
if writer is not None:
    writer.release()
    print(f'Saved annotated video to {OUTPUT_VIDEO_PATH}')
cv2.destroyAllWindows()
if not SHOW_PREVIEW:
    sys.stdout.write('\n')
    sys.stdout.flush()

# Convert the list of data to a CSV and save it
df = pd.DataFrame(results_data)
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f'Saved {OUTPUT_CSV_PATH} successfully!')