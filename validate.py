"""
validate.py — sanity-check a CSV produced by main.py.

Usage:
    python validate.py output_video.csv --video /path/to/video.mp4

The --video flag is optional. When supplied, bbox coordinates are checked
against the actual frame dimensions. Without it, only structural and logical
checks run.

Exit code 0 = all checks passed. Non-zero = at least one check failed.
"""

import argparse
import ast
import sys
from pathlib import Path

import cv2
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = '\u2713'
FAIL = '\u2717'


def report(label: str, ok: bool, detail: str = '') -> bool:
    status = PASS if ok else FAIL
    line = f'  [{status}] {label}'
    if detail:
        line += f' — {detail}'
    print(line)
    return ok


def parse_bbox(raw: str) -> tuple[float, float, float, float] | None:
    """Parse '[cx, cy, w, h]' string into a tuple. Returns None on failure."""
    try:
        values = ast.literal_eval(raw)
        if len(values) == 4:
            return tuple(float(v) for v in values)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_required_columns(df: pd.DataFrame) -> bool:
    required = {'frame_id', 'bbox', 'track_id', 'class', 'timestamp'}
    missing = required - set(df.columns)
    return report(
        'Required columns present',
        not missing,
        f'missing: {missing}' if missing else f'{len(df)} rows',
    )


def check_no_nulls(df: pd.DataFrame) -> bool:
    null_counts = df[['frame_id', 'track_id', 'timestamp']].isnull().sum()
    bad = null_counts[null_counts > 0]
    return report(
        'No missing values in key columns',
        bad.empty,
        str(bad.to_dict()) if not bad.empty else 'clean',
    )


def check_positive_ids(df: pd.DataFrame) -> bool:
    bad_frame = (df['frame_id'] < 0).sum()
    bad_track = (df['track_id'] < 1).sum()
    ok = bad_frame == 0 and bad_track == 0
    detail = []
    if bad_frame:
        detail.append(f'{bad_frame} negative frame_ids')
    if bad_track:
        detail.append(f'{bad_track} track_ids < 1')
    return report('frame_id ≥ 0 and track_id ≥ 1', ok, ', '.join(detail) or 'all valid')


def check_timestamps_non_decreasing(df: pd.DataFrame) -> bool:
    # Timestamps should be non-decreasing when rows are sorted by frame_id.
    # Within a single frame there will be many rows with the same timestamp,
    # so we check per-frame min timestamps are strictly non-decreasing.
    frame_ts = df.groupby('frame_id')['timestamp'].min().sort_index()
    inversions = (frame_ts.diff().dropna() < 0).sum()
    return report(
        'Timestamps non-decreasing across frames',
        inversions == 0,
        f'{inversions} inversions found' if inversions else 'monotonic',
    )


def check_timestamps_match_frame_ids(df: pd.DataFrame) -> bool:
    # timestamp = frame_id / fps so timestamp / frame_id should be constant
    # (ignoring frame_id == 0 to avoid divide-by-zero).
    non_zero = df[df['frame_id'] > 0].copy()
    if non_zero.empty:
        return report('Timestamps match frame_ids', True, 'only frame 0 present, skipped')
    ratios = non_zero['timestamp'] / non_zero['frame_id']
    # Allow 1 ms tolerance for floating-point rounding.
    spread = ratios.max() - ratios.min()
    ok = spread < 0.001
    return report(
        'Timestamps consistent with frame_ids (timestamp = frame_id / fps)',
        ok,
        f'ratio spread={spread:.6f}s (expected ~0)' if not ok else f'fps≈{1/ratios.mean():.3f}',
    )


def check_bbox_parseable(df: pd.DataFrame) -> bool:
    failures = df['bbox'].apply(lambda v: parse_bbox(v) is None).sum()
    return report(
        'All bbox values parseable as [cx, cy, w, h]',
        failures == 0,
        f'{failures} unparseable rows' if failures else 'all valid',
    )


def check_bbox_positive_dimensions(df: pd.DataFrame) -> bool:
    parsed = df['bbox'].apply(parse_bbox)
    widths = parsed.apply(lambda b: b[2] if b else None).dropna()
    heights = parsed.apply(lambda b: b[3] if b else None).dropna()
    bad_w = (widths <= 0).sum()
    bad_h = (heights <= 0).sum()
    ok = bad_w == 0 and bad_h == 0
    detail = []
    if bad_w:
        detail.append(f'{bad_w} non-positive widths')
    if bad_h:
        detail.append(f'{bad_h} non-positive heights')
    return report('bbox width and height > 0', ok, ', '.join(detail) or 'all valid')


def check_bbox_within_frame(df: pd.DataFrame, frame_w: int, frame_h: int) -> bool:
    # bbox is [cx, cy, w, h] — the box edges are cx±w/2 and cy±h/2.
    parsed = df['bbox'].apply(parse_bbox).dropna()
    x_min = parsed.apply(lambda b: b[0] - b[2] / 2)
    x_max = parsed.apply(lambda b: b[0] + b[2] / 2)
    y_min = parsed.apply(lambda b: b[1] - b[3] / 2)
    y_max = parsed.apply(lambda b: b[1] + b[3] / 2)

    # Small tolerance (1 px) for floating-point bbox edges that clip slightly.
    tol = 1.0
    oob = (
        (x_min < -tol) | (x_max > frame_w + tol) |
        (y_min < -tol) | (y_max > frame_h + tol)
    ).sum()
    return report(
        f'bbox coordinates within frame ({frame_w}×{frame_h})',
        oob == 0,
        f'{oob} out-of-bounds boxes' if oob else 'all within bounds',
    )


def check_unique_track_per_frame(df: pd.DataFrame) -> bool:
    # Each (frame_id, track_id) pair must be unique — a tracker should never
    # assign the same ID to two different objects in the same frame.
    dupes = df.duplicated(subset=['frame_id', 'track_id']).sum()
    return report(
        'No duplicate (frame_id, track_id) pairs',
        dupes == 0,
        f'{dupes} duplicates' if dupes else 'all unique',
    )


# ---------------------------------------------------------------------------
# Core validation — importable by main.py or run standalone
# ---------------------------------------------------------------------------

def run_validation(csv_path: str | Path, video_path: str | None = None) -> bool:
    """
    Validate a CSV produced by main.py.

    Returns True if all checks pass, False otherwise.
    Can be imported and called directly without going through argparse.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f'Validation error: {csv_path} not found.')
        return False

    df = pd.read_csv(csv_path)
    print(f'\nValidating {csv_path.name}  ({len(df):,} rows)\n')

    results: list[bool] = []

    print('--- Structure ---')
    results.append(check_required_columns(df))
    results.append(check_no_nulls(df))

    print('\n--- IDs ---')
    results.append(check_positive_ids(df))
    results.append(check_unique_track_per_frame(df))

    print('\n--- Timestamps ---')
    results.append(check_timestamps_non_decreasing(df))
    results.append(check_timestamps_match_frame_ids(df))

    print('\n--- Bounding boxes ---')
    results.append(check_bbox_parseable(df))
    results.append(check_bbox_positive_dimensions(df))

    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if frame_w > 0 and frame_h > 0:
            results.append(check_bbox_within_frame(df, frame_w, frame_h))
        else:
            print(f'  [!] Could not read frame dimensions from {video_path}, skipping bounds check.')
    else:
        print('  [!] --video not provided, skipping bbox bounds check.')

    passed = sum(results)
    total = len(results)
    print(f'\n{passed}/{total} checks passed.')

    return passed == total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description='Validate a main.py output CSV.')
    parser.add_argument('csv', help='Path to the output CSV to validate')
    parser.add_argument(
        '--video',
        default=None,
        help='Source video path; enables bbox bounds check against real frame size',
    )
    args = parser.parse_args()
    return 0 if run_validation(args.csv, args.video) else 1


if __name__ == '__main__':
    sys.exit(main())
