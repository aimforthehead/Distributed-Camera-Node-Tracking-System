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
from tracking import StereoTracker
from triangulate_utils import load_calibration, detect_ball, MotionDetector, check_frame_size


def make_detectors(args, hsv_params):
    """Each detector returns (mask, [candidate centres])."""
    if args.detector == "motion":
        m0, m1 = MotionDetector(min_area=args.min_area), MotionDetector(min_area=args.min_area)
        return (lambda f: m0.candidates(f, k=3)), (lambda f: m1.candidates(f, k=3))

    def color(f):
        mask, c = detect_ball(f, hsv_params)
        return mask, ([c] if c is not None else [])
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
    print(f"Calibration loaded. Baseline = {float(stereo['baseline_mm'][0]):.1f} mm")
    print(f"Detector: {args.detector}")
    print("Q quit | M mask | R reset background | F fullscreen | S screenshot\n")

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    check_frame_size(cap0, stereo["img_size"], f"camera {args.cam0}")
    if "img_size1" in stereo.files:
        check_frame_size(cap1, stereo["img_size1"], f"camera {args.cam1}")
    det0, det1 = make_detectors(args, hsv_params)
    tracker = StereoTracker(stereo, max_reproj=args.max_reproj)
    dash = Dashboard(stereo)
    show_mask = False

    log_fh = open(args.log, "w", newline="")
    writer = csv.writer(log_fh)
    writer.writerow(["timestamp_s", "x_mm", "y_mm", "z_mm", "range_mm", "speed_mps",
                     "closing_mps", "reproj_px", "detected", "sync_lag_ms"])

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
            mask0, cands0 = det0(frame0)
            mask1, cands1 = det1(frame1)
            r = tracker.update(t, cands0, cands1, frame1.shape[1])
            pos, reproj = r["pos"], r["reproj"]
            coasting = r["state"] in ("coasting", "rejected")

            # detected: 1 = measured this frame, 2 = predicted through a short dropout, 0 = none
            flag = "0" if pos is None else ("2" if coasting else "1")
            if pos is not None:
                writer.writerow([f"{t:.4f}", f"{pos[0]:.1f}", f"{pos[1]:.1f}", f"{pos[2]:.1f}",
                                 f"{np.linalg.norm(pos):.1f}", f"{r['speed']:.3f}", f"{r['closing']:.3f}",
                                 "" if reproj is None else f"{reproj:.2f}", flag, f"{r['lag_ms']:.0f}"])
            else:
                writer.writerow([f"{t:.4f}", "", "", "", "", "", "",
                                 "" if reproj is None else f"{reproj:.2f}", "0", f"{r['lag_ms']:.0f}"])

            c0 = r["c0"] or (cands0[0] if cands0 else None)
            c1 = r["c1"] or (cands1[0] if cands1 else None)
            canvas = dash.render(t, frame0, frame1, mask0, mask1, c0, c1, pos, r["speed"],
                                 r["closing"], args.detector, show_mask,
                                 reproj if r["raw"] is not None or r["no_match"] else None,
                                 coasting=coasting, lag_ms=r["lag_ms"])
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
        print(f"Log saved → {args.log}  (camera sync offset {tracker.lag * 1000:+.0f} ms, "
              f"{tracker.glitches} impossible jumps rejected)")


if __name__ == "__main__":
    main()
