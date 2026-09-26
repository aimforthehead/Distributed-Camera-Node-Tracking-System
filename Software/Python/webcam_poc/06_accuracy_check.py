#!/usr/bin/env python3
"""
06_accuracy_check.py — Measure tracking accuracy at known ground-truth positions.

Procedure per test point:
  1. Place the ball at a known position.
  2. Press SPACE — the script captures N frames and averages the 3D result.
  3. When prompted, enter the ground-truth X Y Z you measured in mm.
  4. Repeat for as many points as you like, then press Q to see the summary.

How to measure ground truth:
  - Camera 0's optical centre is the world origin (+Z forward, +X right, +Y down).
  - A practical shortcut: place the ball at a point you can measure with a ruler
    from camera 0's lens: e.g., 80 cm in front and 30 cm to the right at lens height
    → X=300, Y=0, Z=800 mm.

Usage:
    python 06_accuracy_check.py
    python 06_accuracy_check.py --cam0 0 --cam1 1 --n_frames 50

Controls:
    SPACE — capture measurement at current ball position
    Q     — quit and print summary
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from triangulate_utils import load_calibration, detect_ball, triangulate_point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    parser.add_argument("--n_frames", type=int, default=40,
                        help="Frames to average per measurement (default: 40)")
    args = parser.parse_args()

    stereo, hsv_params = load_calibration(args.calib_dir)
    K0, dist0 = stereo["K0"], stereo["dist0"]
    K1, dist1 = stereo["K1"], stereo["dist1"]
    P0, P1    = stereo["P0"], stereo["P1"]

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)

    measurements: list[dict] = []

    print("=" * 60)
    print("ACCURACY CHECK")
    print("=" * 60)
    print("Coord system: cam0 = origin, +Z forward, +X right, +Y down, units mm")
    print(f"Each capture averages {args.n_frames} frames.")
    print("SPACE = capture | Q = quit + summary\n")

    while True:
        cap0.grab()
        cap1.grab()
        ok0, frame0 = cap0.retrieve()
        ok1, frame1 = cap1.retrieve()
        if not ok0 or not ok1:
            break

        _, center0 = detect_ball(frame0, hsv_params)
        _, center1 = detect_ball(frame1, hsv_params)
        both = center0 is not None and center1 is not None

        for frame, center, label in [(frame0, center0, "Cam0"), (frame1, center1, "Cam1")]:
            status = "DETECTED" if center else "not found"
            color = (0, 220, 0) if center else (0, 0, 220)
            cv2.putText(frame, f"{label}: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame,
                        f"pts: {len(measurements)}  |  SPACE=capture  Q=quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
            if both:
                cv2.putText(frame, "Ready — press SPACE", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)

        h = min(frame0.shape[0], frame1.shape[0], 480)
        def rh(f: np.ndarray) -> np.ndarray:
            s = h / f.shape[0]
            return cv2.resize(f, (int(f.shape[1] * s), h))

        cv2.imshow("Step 6 — Accuracy Check  |  SPACE=capture  Q=quit",
                   np.hstack([rh(frame0), rh(frame1)]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print(f"\nCapturing {args.n_frames} frames — hold the ball still ...")
            samples: list[np.ndarray] = []
            missed = 0
            while len(samples) < args.n_frames:
                cap0.grab(); cap1.grab()
                _, f0 = cap0.retrieve(); _, f1 = cap1.retrieve()
                _, c0 = detect_ball(f0, hsv_params)
                _, c1 = detect_ball(f1, hsv_params)
                if c0 is not None and c1 is not None:
                    pt = triangulate_point(c0, c1, K0, dist0, K1, dist1, P0, P1)
                    samples.append(pt)
                    print(f"\r  {len(samples)}/{args.n_frames}  "
                          f"({pt[0]:.0f}, {pt[1]:.0f}, {pt[2]:.0f}) mm", end="", flush=True)
                else:
                    missed += 1
                    if missed > args.n_frames * 5:
                        print("\nWARNING: ball not detected consistently — aborting capture.")
                        samples = []
                        break
            print()
            if not samples:
                continue

            mean_pos = np.mean(samples, axis=0)
            std_pos  = np.std(samples, axis=0)
            print(f"  Measured : X={mean_pos[0]:.1f}  Y={mean_pos[1]:.1f}  Z={mean_pos[2]:.1f} mm")
            print(f"  Std dev  : X={std_pos[0]:.1f}   Y={std_pos[1]:.1f}   Z={std_pos[2]:.1f} mm")

            gt_str = input("  Ground truth X Y Z in mm (e.g. 300 0 800): ").strip()
            try:
                gt_vals = list(map(float, gt_str.split()))
                if len(gt_vals) != 3:
                    raise ValueError
            except ValueError:
                print("  Invalid input — skipping this point.")
                continue

            gt = np.array(gt_vals)
            err = mean_pos - gt
            dist_err = float(np.linalg.norm(err))
            print(f"  Error    : dX={err[0]:.1f}  dY={err[1]:.1f}  dZ={err[2]:.1f} mm")
            print(f"  3D dist  : {dist_err:.1f} mm")
            measurements.append({
                "ground_truth_mm": gt.tolist(),
                "measured_mm": mean_pos.tolist(),
                "std_mm": std_pos.tolist(),
                "error_xyz_mm": err.tolist(),
                "distance_error_mm": dist_err,
                "n_frames": len(samples),
            })

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    if not measurements:
        print("No measurements recorded.")
        return

    errors = [m["distance_error_mm"] for m in measurements]
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    print(f"{'#':>3}  {'GT (mm)':>22}  {'Measured (mm)':>22}  {'Error':>8}")
    for i, m in enumerate(measurements):
        gt = m["ground_truth_mm"]
        me = m["measured_mm"]
        print(f"{i+1:>3}  ({gt[0]:6.0f},{gt[1]:6.0f},{gt[2]:6.0f})  "
              f"({me[0]:6.0f},{me[1]:6.0f},{me[2]:6.0f})  {m['distance_error_mm']:7.1f} mm")
    print(f"\n  Mean error : {np.mean(errors):.1f} mm")
    print(f"  Std        : {np.std(errors):.1f} mm")
    print(f"  Min / Max  : {np.min(errors):.1f} / {np.max(errors):.1f} mm")

    out = Path("accuracy_results.json")
    with open(out, "w") as f:
        json.dump(measurements, f, indent=2)
    print(f"\nFull results saved → {out}")


if __name__ == "__main__":
    main()
