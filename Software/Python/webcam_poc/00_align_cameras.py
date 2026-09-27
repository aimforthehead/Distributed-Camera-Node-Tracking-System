#!/usr/bin/env python3
"""
00_align_cameras.py — Aim both cameras at the same point before calibrating.

Put a target at the centre of the flight zone: a tape "X" on the wall, or the
phone showing the checkerboard. Each camera view shows a crosshair and tells you
exactly how many degrees to turn/tilt that camera until its centre is on the target.
When both say ON TARGET, the cameras converge on the flight zone at the correct angle
for whatever spacing you chose.

The target is found automatically if it is the checkerboard; otherwise click on it
in each camera view.

Usage:
    python 00_align_cameras.py --cam0 1 --cam1 2 --baseline-cm 40 --distance-cm 150

    --baseline-cm   distance between the two lenses
    --distance-cm   distance from the midpoint between the cameras to the target

Controls:
    click — mark the target in that camera view (overrides the checkerboard)
    C     — clear clicked targets
    Q     — quit
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import cv2
import numpy as np

ON_TARGET_DEG = 1.0
WIN = "Step 0 — Camera Alignment  |  click target  C=clear  Q=quit"


def parse_board(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    return int(parts[0]), int(parts[1])


def aim_angles(pt: tuple[float, float], K: np.ndarray, dist: np.ndarray) -> tuple[float, float]:
    """Pan (+ = target right) and tilt (+ = target below) of pt from the optical axis, degrees."""
    n = cv2.undistortPoints(np.array([[pt]], np.float32), K, dist).ravel()
    return math.degrees(math.atan(n[0])), math.degrees(math.atan(n[1]))


def load_intrinsics(calib_dir: Path, idx: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray, bool]:
    f = calib_dir / f"cam{idx}_intrinsics.npz"
    if f.exists():
        d = np.load(f)
        return d["K"], d["dist"], True
    fx = (w / 2) / math.tan(math.radians(30))  # assume ~60° horizontal field of view
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]]), np.zeros(5), False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--baseline-cm", type=float, default=40.0)
    parser.add_argument("--distance-cm", type=float, default=150.0)
    parser.add_argument("--board", type=parse_board, default=(9, 6), metavar="COLSxROWS")
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    args = parser.parse_args()

    toe_in = math.degrees(math.atan((args.baseline_cm / 2) / args.distance_cm))
    print(f"Cameras {args.baseline_cm:g} cm apart, target {args.distance_cm:g} cm away:")
    print(f"  each camera turns ~{toe_in:.1f}° inward (total convergence {2 * toe_in:.1f}°)")
    print(f"  both lenses at the same height, target at that height too")
    print("Rotate each camera until its view says ON TARGET.\n")

    caps = [cv2.VideoCapture(args.cam0), cv2.VideoCapture(args.cam1)]
    idxs = [args.cam0, args.cam1]
    intr: list = [None, None]
    clicks: list = [None, None]
    layout = {"w0": 1, "h": 1, "s": [1.0, 1.0]}

    def on_mouse(event, x, y, _f, _p):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        i = 0 if x < layout["w0"] else 1
        xo = x if i == 0 else x - layout["w0"]
        clicks[i] = (xo / layout["s"][i], y / layout["s"][i])

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        for c in caps:
            c.grab()
        frames = [c.retrieve()[1] for c in caps]
        if any(f is None for f in frames):
            print("ERROR: failed to read frames")
            break

        views = []
        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            if intr[i] is None:
                intr[i] = load_intrinsics(args.calib_dir, idxs[i], w, h)
                if not intr[i][2]:
                    print(f"cam{idxs[i]}: no calibration file yet — angles are approximate")
            K, dist, exact = intr[i]

            target = clicks[i]
            if target is None:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(g, args.board, None, cv2.CALIB_CB_FAST_CHECK)
                if found:
                    target = tuple(corners.reshape(-1, 2).mean(axis=0))

            v = f.copy()
            cx, cy = int(K[0, 2]), int(K[1, 2])
            color = (0, 0, 230)
            lines = [f"CAM {idxs[i]}: no target - show checkerboard or click the mark"]
            if target is not None:
                pan, tilt = aim_angles(target, K, dist)
                ok = abs(pan) < ON_TARGET_DEG and abs(tilt) < ON_TARGET_DEG
                color = (0, 220, 0) if ok else (0, 200, 255)
                tx, ty = int(target[0]), int(target[1])
                cv2.arrowedLine(v, (cx, cy), (tx, ty), color, 3, cv2.LINE_AA, tipLength=0.08)
                cv2.circle(v, (tx, ty), 12, color, 3, cv2.LINE_AA)
                if ok:
                    lines = [f"CAM {idxs[i]}: ON TARGET"]
                else:
                    lines = [f"CAM {idxs[i]}: turn {'RIGHT' if pan > 0 else 'LEFT'} {abs(pan):.1f} deg",
                             f"tilt {'DOWN' if tilt > 0 else 'UP'} {abs(tilt):.1f} deg"]
                lines.append(f"(off by pan {pan:+.1f}, tilt {tilt:+.1f}"
                             f"{'' if exact else ', approx'})")

            arm = max(20, w // 30)
            cv2.line(v, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 2)
            cv2.line(v, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 2)
            scale = max(0.8, w / 1280)
            for k, txt in enumerate(lines):
                cv2.putText(v, txt, (15, int(45 * scale) + k * int(42 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale if k == 0 else 0.75 * scale,
                            color, 2, cv2.LINE_AA)
            views.append(v)

        h = min(views[0].shape[0], views[1].shape[0], 480)
        s = [h / views[0].shape[0], h / views[1].shape[0]]
        resized = [cv2.resize(v, (int(v.shape[1] * s[k]), h)) for k, v in enumerate(views)]
        layout.update(w0=resized[0].shape[1], h=h, s=s)
        banner = np.full((34, resized[0].shape[1] + resized[1].shape[1], 3), 25, np.uint8)
        cv2.putText(banner, f"Aim both crosshairs at the same point.  Recommended: each camera "
                    f"~{toe_in:.1f} deg inward, lenses at equal height.",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.imshow(WIN, np.vstack([np.hstack(resized), banner]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            clicks[0] = clicks[1] = None

    for c in caps:
        c.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
