#!/usr/bin/env python3
"""
05_triangulate.py — Live 3D tracking pipeline.

Loads calibration from steps 02/03 and HSV params from step 04.
Grabs synchronised frames, detects the ball in both views via HSV thresholding,
triangulates to 3D using the DLT method, and shows:
  • Both camera feeds with detection overlay
  • Live matplotlib 3D scatter plot (updates at ~20 Hz)

Positions are logged to tracking_log.csv.

Usage:
    python 05_triangulate.py
    python 05_triangulate.py --cam0 0 --cam1 1

Controls (OpenCV window):
    Q — quit

Coordinate system:
    Origin = camera 0 optical centre
    +X = cam0 right, +Y = cam0 DOWN, +Z = cam0 forward (into scene)
    Units = mm (set by checkerboard square size during calibration)

    The 3D plot inverts Y so that "up" visually points up.
"""
import argparse
import csv
import sys
import time
from pathlib import Path
import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass  # fall back to whatever backend is available

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers projection)

from triangulate_utils import load_calibration, detect_ball, triangulate_point


MAX_HISTORY = 120  # trail length in 3D plot
PLOT_HZ = 20       # max plot refresh rate


def setup_3d_plot() -> tuple:
    plt.ion()
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm) — depth")
    ax.set_zlabel("−Y (mm) — up")
    ax.set_title("Live 3D Tracking")
    fig.tight_layout()
    return fig, ax


def update_plot(ax, history: list, current: np.ndarray | None) -> None:
    ax.cla()
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm) — depth")
    ax.set_zlabel("−Y (mm) — up")
    ax.set_title("Live 3D Tracking")

    ax.scatter([0], [0], [0], c="royalblue", s=120, marker="^", label="Cam0 origin", zorder=5)

    if history:
        xs = [p[0] for p in history]
        zs = [p[2] for p in history]
        ys = [-p[1] for p in history]
        ax.plot(xs, zs, ys, color="lightgray", linewidth=1, alpha=0.6)

    if current is not None:
        ax.scatter([current[0]], [current[2]], [-current[1]],
                   c="crimson", s=180, zorder=6, label="Object")
        ax.text(current[0], current[2], -current[1],
                f"  ({current[0]:.0f}, {current[1]:.0f}, {current[2]:.0f}) mm",
                fontsize=8)

    ax.legend(loc="upper left")
    plt.draw()
    plt.pause(0.001)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    parser.add_argument("--log", type=Path, default=Path("tracking_log.csv"))
    args = parser.parse_args()

    stereo, hsv_params = load_calibration(args.calib_dir)
    K0, dist0 = stereo["K0"], stereo["dist0"]
    K1, dist1 = stereo["K1"], stereo["dist1"]
    P0, P1    = stereo["P0"], stereo["P1"]
    baseline  = float(stereo["baseline_mm"][0])

    print(f"Calibration loaded.  Baseline = {baseline:.1f} mm")
    print(f"HSV params: {hsv_params}")
    print("Q in the camera window to quit.\n")

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)

    fig, ax = setup_3d_plot()
    history: list[np.ndarray] = []
    last_plot = 0.0

    log_fh = open(args.log, "w", newline="")
    writer = csv.writer(log_fh)
    writer.writerow(["timestamp_s", "x_mm", "y_mm", "z_mm", "detected"])

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
            _, center0 = detect_ball(frame0, hsv_params)
            _, center1 = detect_ball(frame1, hsv_params)

            pos3d: np.ndarray | None = None
            if center0 is not None and center1 is not None:
                pos3d = triangulate_point(center0, center1, K0, dist0, K1, dist1, P0, P1)
                history.append(pos3d)
                if len(history) > MAX_HISTORY:
                    history.pop(0)
                writer.writerow([f"{t:.4f}", f"{pos3d[0]:.1f}",
                                 f"{pos3d[1]:.1f}", f"{pos3d[2]:.1f}", "1"])
            else:
                writer.writerow([f"{t:.4f}", "", "", "", "0"])

            # Camera view overlays
            for frame, center, label in [(frame0, center0, "Cam0"), (frame1, center1, "Cam1")]:
                if center:
                    cx, cy = int(center[0]), int(center[1])
                    cv2.circle(frame, (cx, cy), 22, (0, 230, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 230, 0), -1)
                status = "TRACKING" if center else "searching"
                color = (0, 220, 0) if center else (0, 60, 220)
                cv2.putText(frame, f"{label}: {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if pos3d is not None and label == "Cam0":
                    cv2.putText(frame,
                                f"3D: ({pos3d[0]:.0f}, {pos3d[1]:.0f}, {pos3d[2]:.0f}) mm",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)

            h = min(frame0.shape[0], frame1.shape[0], 480)
            def rh(f: np.ndarray) -> np.ndarray:
                s = h / f.shape[0]
                return cv2.resize(f, (int(f.shape[1] * s), h))

            cv2.imshow("Step 5 — Tracking  |  Q=quit", np.hstack([rh(frame0), rh(frame1)]))

            if t - last_plot > 1.0 / PLOT_HZ:
                update_plot(ax, history, pos3d)
                last_plot = t

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
