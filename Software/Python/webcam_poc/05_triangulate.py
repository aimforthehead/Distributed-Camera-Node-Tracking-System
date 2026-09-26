#!/usr/bin/env python3
"""
05_triangulate.py — Live 3D tracking pipeline.

Grabs synchronised frames from both cameras, detects the target in each view,
triangulates to 3D (DLT), and shows:
  • Both camera feeds with detection overlay, 3D position, range and closing speed
  • Live matplotlib 3D plot with the target's trail

Two detectors:
  --detector motion   (default) Any moving object against a static background.
                      Background subtraction: pixels that change between frames
                      are flagged as a moving target. Use for a drone moved by hand.
                      Needs no colour tuning; keep the cameras perfectly still.
  --detector color    HSV colour threshold (needs calibration/hsv_params.json from step 04).

Usage:
    python 05_triangulate.py --cam0 1 --cam1 2
    python 05_triangulate.py --cam0 1 --cam1 2 --detector color
    python 05_triangulate.py --cam0 1 --cam1 2 --min-area 200   # ignore small noise

Controls (camera window):
    Q — quit
    M — toggle motion-mask view (shows what the detector sees)
    R — reset background model (after lighting changes)

Coordinate system:
    Origin = camera 0 optical centre
    +X = cam0 right, +Y = cam0 DOWN, +Z = cam0 forward (into scene)
    Units = mm (set by checkerboard square size during calibration)
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from triangulate_utils import load_calibration, detect_ball, triangulate_point, MotionDetector


MAX_HISTORY = 120   # trail length in 3D plot
PLOT_HZ = 10        # max plot refresh rate
SMOOTHING = 0.3     # EMA weight for speed estimates


def setup_3d_plot():
    plt.ion()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def update_plot(ax, history: list, current: np.ndarray | None, title: str) -> None:
    ax.cla()
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm) — depth")
    ax.set_zlabel("−Y (mm) — up")
    ax.set_title(title)
    ax.scatter([0], [0], [0], c="royalblue", s=120, marker="^", label="Cam0 (origin)")
    if history:
        h = np.array(history)
        ax.plot(h[:, 0], h[:, 2], -h[:, 1], color="gray", linewidth=1, alpha=0.6)
    if current is not None:
        ax.scatter([current[0]], [current[2]], [-current[1]], c="crimson", s=180, label="Target")
    ax.legend(loc="upper left")
    plt.draw()
    plt.pause(0.001)


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
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    parser.add_argument("--log", type=Path, default=Path("tracking_log.csv"))
    args = parser.parse_args()

    stereo, hsv_params = load_calibration(args.calib_dir, need_hsv=args.detector == "color")
    K0, dist0 = stereo["K0"], stereo["dist0"]
    K1, dist1 = stereo["K1"], stereo["dist1"]
    P0, P1 = stereo["P0"], stereo["P1"]
    print(f"Calibration loaded. Baseline = {float(stereo['baseline_mm'][0]):.1f} mm")
    print(f"Detector: {args.detector}")
    print("Q = quit | M = show motion mask | R = reset background\n")

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    det0, det1 = make_detectors(args, hsv_params)

    fig, ax = setup_3d_plot()
    history: list[np.ndarray] = []
    last_plot = 0.0
    show_mask = False
    prev: tuple[float, np.ndarray] | None = None
    speed = closing = 0.0

    log_fh = open(args.log, "w", newline="")
    writer = csv.writer(log_fh)
    writer.writerow(["timestamp_s", "x_mm", "y_mm", "z_mm", "range_mm",
                     "speed_mps", "closing_mps", "detected"])

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
            if center0 is not None and center1 is not None:
                pos3d = triangulate_point(center0, center1, K0, dist0, K1, dist1, P0, P1)
                # A point behind the camera means the two detections weren't the same object
                if pos3d[2] <= 0:
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
                history.append(pos3d)
                if len(history) > MAX_HISTORY:
                    history.pop(0)
                writer.writerow([f"{t:.4f}", f"{pos3d[0]:.1f}", f"{pos3d[1]:.1f}",
                                 f"{pos3d[2]:.1f}", f"{rng:.1f}", f"{speed:.3f}",
                                 f"{closing:.3f}", "1"])
            else:
                prev = None
                writer.writerow([f"{t:.4f}", "", "", "", "", "", "", "0"])

            views = []
            for frame, mask, center, label in [(frame0, mask0, center0, "Cam0"),
                                               (frame1, mask1, center1, "Cam1")]:
                view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask and mask is not None else frame
                if center:
                    cx, cy = int(center[0]), int(center[1])
                    cv2.circle(view, (cx, cy), 22, (0, 230, 0), 2)
                    cv2.circle(view, (cx, cy), 4, (0, 230, 0), -1)
                status = "TARGET" if center else "searching"
                color = (0, 220, 0) if center else (0, 60, 220)
                cv2.putText(view, f"{label}: {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                views.append(view)

            if pos3d is not None:
                lines = [
                    f"3D: ({pos3d[0]:.0f}, {pos3d[1]:.0f}, {pos3d[2]:.0f}) mm",
                    f"Range: {np.linalg.norm(pos3d) / 1000:.2f} m",
                    f"Speed: {speed:.2f} m/s",
                    f"{'APPROACHING' if closing > 0.05 else 'Receding' if closing < -0.05 else 'Holding'}"
                    f" {abs(closing):.2f} m/s",
                ]
                for i, txt in enumerate(lines):
                    cv2.putText(views[0], txt, (10, 70 + 34 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)

            h = min(views[0].shape[0], views[1].shape[0], 480)
            resized = [cv2.resize(v, (int(v.shape[1] * h / v.shape[0]), h)) for v in views]
            cv2.imshow("Step 5 — Tracking  |  Q=quit  M=mask  R=reset", np.hstack(resized))

            if t - last_plot > 1.0 / PLOT_HZ:
                title = "Live 3D Tracking"
                if pos3d is not None:
                    title += f" — range {np.linalg.norm(pos3d) / 1000:.2f} m, {speed:.2f} m/s"
                update_plot(ax, history, pos3d, title)
                last_plot = t

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("r") and args.detector == "motion":
                det0, det1 = make_detectors(args, hsv_params)
                print("Background model reset.")

    finally:
        log_fh.close()
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()
        print(f"Log saved → {args.log}")


if __name__ == "__main__":
    main()
