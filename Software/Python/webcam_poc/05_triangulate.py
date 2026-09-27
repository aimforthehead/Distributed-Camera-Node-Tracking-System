#!/usr/bin/env python3
"""
05_triangulate.py — Live 3D tracking with the operator console.

Grabs synchronised frames from both cameras, detects the target in each view,
triangulates to 3D (DLT), and shows everything in one console window:
  • Both camera feeds with target lock brackets
  • Top-down radar view: camera positions, coverage, target and its trail
  • Telemetry: X / height / forward, range, speed, closing speed, bearing, elevation
  • Range-over-time graph and an APPROACHING alert

Two detectors:
  --detector motion   (default) Any moving object against a static background.
                      Pixels that change from the learned background are flagged
                      as a moving target. Use for a drone moved by hand.
                      Needs no colour tuning; keep the cameras perfectly still.
  --detector color    HSV colour threshold (needs calibration/hsv_params.json from step 04).

Usage:
    python 05_triangulate.py --cam0 0 --cam1 1
    python 05_triangulate.py --cam0 0 --cam1 1 --min-area 300   # ignore small noise
    python 05_triangulate.py --cam0 0 --cam1 1 --detector color

Controls (console window):
    Q — quit
    M — toggle motion-mask view (shows what the detector sees)
    R — reset background model (after lighting changes)
    F — toggle fullscreen
    S — save a screenshot PNG (for the pitch deck)

Coordinate system:
    Origin = camera 0 optical centre
    +X = cam0 right, +Y = cam0 DOWN, +Z = cam0 forward (into scene)
    Units = mm internally (set by checkerboard square size); shown in metres
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import cv2
import numpy as np

from dashboard import Dashboard
from triangulate_utils import (load_calibration, detect_ball, triangulate_point,
                               reprojection_error, MotionDetector, check_frame_size)


SMOOTHING = 0.3  # EMA weight for speed estimates


def make_detectors(args, hsv_params):
    if args.detector == "motion":
        return (MotionDetector(min_area=args.min_area),
                MotionDetector(min_area=args.min_area))
    color = lambda f: detect_ball(f, hsv_params)  # noqa: E731
    return color, color


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--detector", choices=["motion", "color"], default="motion")
    parser.add_argument("--min-area", type=int, default=80,
                        help="Smallest moving blob in pixels (motion mode)")
    parser.add_argument("--max-reproj", type=float, default=25.0,
                        help="Reject 3D points whose reprojection error exceeds this (px)")
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    parser.add_argument("--log", type=Path, default=Path("tracking_log.csv"))
    args = parser.parse_args()

    stereo, hsv_params = load_calibration(args.calib_dir, need_hsv=args.detector == "color")
    K0, dist0 = stereo["K0"], stereo["dist0"]
    K1, dist1 = stereo["K1"], stereo["dist1"]
    P0, P1 = stereo["P0"], stereo["P1"]
    print(f"Calibration loaded. Baseline = {float(stereo['baseline_mm'][0]):.1f} mm")
    print(f"Detector: {args.detector}")
    print("Q quit | M mask | R reset background | F fullscreen | S screenshot\n")

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    check_frame_size(cap0, stereo["img_size"], f"camera {args.cam0}")
    if "img_size1" in stereo.files:
        check_frame_size(cap1, stereo["img_size1"], f"camera {args.cam1}")
    det0, det1 = make_detectors(args, hsv_params)
    dash = Dashboard(stereo)

    show_mask = False
    prev: tuple[float, np.ndarray] | None = None
    speed = closing = 0.0

    log_fh = open(args.log, "w", newline="")
    writer = csv.writer(log_fh)
    writer.writerow(["timestamp_s", "x_mm", "y_mm", "z_mm", "range_mm",
                     "speed_mps", "closing_mps", "reproj_px", "detected"])

    try:
        while True:
            cap0.grab()
            cap1.grab()
            ok0, frame0 = cap0.retrieve()
            ok1, frame1 = cap1.retrieve()
            if not ok0 or not ok1:
                print("ERROR: failed to read frames — exiting.")
                break

            t = time.time()
            mask0, center0 = det0(frame0)
            mask1, center1 = det1(frame1)

            pos3d = None
            reproj = None
            if center0 is not None and center1 is not None:
                pos3d = triangulate_point(center0, center1, K0, dist0, K1, dist1, P0, P1)
                reproj = reprojection_error(pos3d, center0, center1, stereo)
                # Behind the camera, or not landing on both detections: the two
                # cameras locked onto different objects, so don't report a position.
                if pos3d[2] <= 0 or reproj > args.max_reproj:
                    pos3d = None

            if pos3d is not None:
                rng = float(np.linalg.norm(pos3d))
                if prev is not None and t - prev[0] > 1e-3:
                    dt = t - prev[0]
                    inst_speed = np.linalg.norm(pos3d - prev[1]) / dt / 1000.0
                    inst_closing = (np.linalg.norm(prev[1]) - rng) / dt / 1000.0
                    speed = (1 - SMOOTHING) * speed + SMOOTHING * inst_speed
                    closing = (1 - SMOOTHING) * closing + SMOOTHING * inst_closing
                prev = (t, pos3d)
                writer.writerow([f"{t:.4f}", f"{pos3d[0]:.1f}", f"{pos3d[1]:.1f}",
                                 f"{pos3d[2]:.1f}", f"{rng:.1f}", f"{speed:.3f}",
                                 f"{closing:.3f}", f"{reproj:.2f}", "1"])
            else:
                prev = None
                speed = closing = 0.0
                writer.writerow([f"{t:.4f}", "", "", "", "", "", "",
                                 "" if reproj is None else f"{reproj:.2f}", "0"])

            canvas = dash.render(t, frame0, frame1, mask0, mask1, center0, center1,
                                 pos3d, speed, closing, args.detector, show_mask, reproj)
            cv2.imshow(Dashboard.WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("r") and args.detector == "motion":
                det0, det1 = make_detectors(args, hsv_params)
                print("Background model reset.")
            elif key == ord("f"):
                dash.toggle_fullscreen()
            elif key == ord("s"):
                name = time.strftime("console_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, canvas)
                print(f"Screenshot saved → {name}")

    finally:
        log_fh.close()
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()
        print(f"Log saved → {args.log}")


if __name__ == "__main__":
    main()
