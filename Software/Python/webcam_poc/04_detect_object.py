#!/usr/bin/env python3
"""
04_detect_object.py — Interactive HSV color tuner for ball detection.

Click on the ball in either camera feed to auto-sample its HSV range.
Adjust the six sliders to fine-tune.  Detection overlay is shown live
on both cameras.  Press S to save, Q to quit.

Usage:
    python 04_detect_object.py
    python 04_detect_object.py --cam0 0 --cam1 1

Output: calibration/hsv_params.json  →  h_low/high, s_low/high, v_low/high
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from triangulate_utils import detect_ball

# Mutable global so slider callbacks can update it
_params: dict[str, int] = {
    "h_low": 0, "h_high": 30, "s_low": 100, "s_high": 255, "v_low": 80, "v_high": 255,
}
_click: list[tuple[int, int] | None] = [None]
_frame_widths: list[int] = [640, 640]  # updated at runtime for click remapping


def _on_click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        _click[0] = (x, y)


def _sample_color(frame: np.ndarray, cx: int, cy: int, win: str) -> None:
    """Sample HSV in a 20×20 px neighbourhood and update _params + sliders."""
    margin = 10
    roi = frame[max(0, cy - margin):cy + margin, max(0, cx - margin):cx + margin]
    if roi.size == 0:
        return
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = int(np.median(hsv_roi[:, :, 0]))
    s = int(np.median(hsv_roi[:, :, 1]))
    v = int(np.median(hsv_roi[:, :, 2]))
    dh, ds, dv = 15, 70, 70
    _params["h_low"]  = max(0,   h - dh)
    _params["h_high"] = min(179, h + dh)
    _params["s_low"]  = max(0,   s - ds)
    _params["s_high"] = min(255, s + ds)
    _params["v_low"]  = max(0,   v - dv)
    _params["v_high"] = min(255, v + dv)
    for key, bar in [("h_low", "H low"), ("h_high", "H high"),
                     ("s_low", "S low"), ("s_high", "S high"),
                     ("v_low", "V low"), ("v_high", "V high")]:
        cv2.setTrackbarPos(bar, win, _params[key])
    print(f"  Sampled HSV centre ({h},{s},{v})  →  "
          f"H[{_params['h_low']}–{_params['h_high']}] "
          f"S[{_params['s_low']}–{_params['s_high']}] "
          f"V[{_params['v_low']}–{_params['v_high']}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    args = parser.parse_args()

    params_file = args.calib_dir / "hsv_params.json"
    if params_file.exists():
        with open(params_file) as f:
            _params.update(json.load(f))
        print(f"Loaded existing HSV params from {params_file}")

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)

    WIN = "Step 4 — HSV Tuner  |  click ball  S=save  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _on_click)

    def _noop(v: int) -> None:
        pass

    for bar, key, maxv in [
        ("H low",  "h_low",  179), ("H high", "h_high", 179),
        ("S low",  "s_low",  255), ("S high", "s_high", 255),
        ("V low",  "v_low",  255), ("V high", "v_high", 255),
    ]:
        cv2.createTrackbar(bar, WIN, _params[key], maxv, _noop)

    # Keep sliders in sync with _params (slider callbacks update these directly)
    def make_cb(k: str):
        def cb(v: int): _params[k] = v
        return cb

    cv2.setTrackbarMin = lambda *_: None  # guard for older OpenCV
    for bar, key in [("H low", "h_low"), ("H high", "h_high"),
                     ("S low", "s_low"), ("S high", "s_high"),
                     ("V low", "v_low"), ("V high", "v_high")]:
        cv2.setTrackbarPos(bar, WIN, _params[key])

    # Re-wire callbacks now so slider movements update _params
    for bar, key, maxv in [
        ("H low",  "h_low",  179), ("H high", "h_high", 179),
        ("S low",  "s_low",  255), ("S high", "s_high", 255),
        ("V low",  "v_low",  255), ("V high", "v_high", 255),
    ]:
        cv2.createTrackbar(bar, WIN, _params[key], maxv, make_cb(key))

    print("Click on the ball in either camera feed to auto-set HSV range.")
    print("S = save params, Q = quit.")

    while True:
        cap0.grab()
        cap1.grab()
        ok0, frame0 = cap0.retrieve()
        ok1, frame1 = cap1.retrieve()
        if not ok0 or not ok1:
            break

        _frame_widths[0] = frame0.shape[1]
        _frame_widths[1] = frame1.shape[1]

        # Handle click
        if _click[0] is not None:
            cx, cy = _click[0]
            _click[0] = None
            # The combined display image has cam0 on the left, cam1 on the right.
            # Work out which camera was clicked and remap coords back to original frame.
            h_disp = min(frame0.shape[0], frame1.shape[0], 480)
            w0_disp = int(frame0.shape[1] * h_disp / frame0.shape[0])
            if cx < w0_disp:
                scale = frame0.shape[0] / h_disp
                _sample_color(frame0, int(cx * scale), int(cy * scale), WIN)
            else:
                scale = frame1.shape[0] / h_disp
                _sample_color(frame1, int((cx - w0_disp) * scale), int(cy * scale), WIN)

        _, center0 = detect_ball(frame0, _params)
        _, center1 = detect_ball(frame1, _params)

        for frame, center, label in [(frame0, center0, "Cam0"), (frame1, center1, "Cam1")]:
            if center:
                cv2.circle(frame, (int(center[0]), int(center[1])), 22, (0, 230, 0), 2)
                cv2.circle(frame, (int(center[0]), int(center[1])), 4, (0, 230, 0), -1)
                cv2.putText(frame, f"({int(center[0])}, {int(center[1])})",
                            (int(center[0]) + 12, int(center[1]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1)
            status = "DETECTED" if center else "not detected"
            color = (0, 220, 0) if center else (0, 0, 220)
            cv2.putText(frame, f"{label}: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        h = min(frame0.shape[0], frame1.shape[0], 480)
        def rh(f: np.ndarray) -> np.ndarray:
            s = h / f.shape[0]
            return cv2.resize(f, (int(f.shape[1] * s), h))

        cv2.imshow(WIN, np.hstack([rh(frame0), rh(frame1)]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            args.calib_dir.mkdir(parents=True, exist_ok=True)
            with open(params_file, 'w') as f:
                json.dump(_params, f, indent=2)
            print(f"Saved → {params_file}")

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
