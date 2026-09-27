#!/usr/bin/env python3
"""
03b_calibrate_with_motion.py — Stereo calibration from any moving object (no checkerboard).

Alternative to step 03. Fly the helicopter/drone (or wave any small object) slowly
through the space both cameras see. Each moment it is seen by both cameras gives one
matched point pair; from a few hundred pairs the script solves how camera 1 sits
relative to camera 0 (essential-matrix method). The only physical measurement needed
is the lens-to-lens distance, measured with a ruler, which sets the real-world scale.

This is how real rooftop nodes would be calibrated in the field: no calibration
pattern, just a drone flown through the shared view.

Still requires each camera's lens calibration from step 02 (checkerboard, close-up).

Usage:
    python 03b_calibrate_with_motion.py --cam0 0 --cam1 1 --baseline-mm 400

How to move the target for a good result:
    • slowly (fast motion + low frame rate = mismatched positions)
    • all over both views: left/right, up/down, AND near/far
    • aim for 200+ samples and 12+/16 coverage in both cameras

Controls:
    C — compute (needs ≥ 60 samples)
    X — clear samples      R — reset background      Q — quit
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import cv2
import numpy as np

from triangulate_utils import (MotionDetector, triangulate_point, reprojection_error,
                               print_rig_geometry)

WARMUP_FRAMES = 30
GRID = 4


def solve_from_matches(p0: np.ndarray, p1: np.ndarray, K0, d0, K1, d1, baseline_mm: float) -> dict:
    """Relative pose of camera 1 from matched pixel points, scaled by the measured baseline."""
    n0 = cv2.undistortPoints(p0.reshape(-1, 1, 2).astype(np.float64), K0, d0).reshape(-1, 2)
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2).astype(np.float64), K1, d1).reshape(-1, 2)
    f = (K0[0, 0] + K0[1, 1] + K1[0, 0] + K1[1, 1]) / 4
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.RANSAC,
                                   prob=0.999, threshold=2.0 / f)
    if E is None:
        raise ValueError("could not estimate camera geometry — collect more varied samples")
    E = E[:3]
    _, R, t, mask = cv2.recoverPose(E, n0, n1, np.eye(3), mask=mask)
    inliers = mask.ravel() > 0
    T = t.reshape(3, 1) / np.linalg.norm(t) * baseline_mm

    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K1 @ np.hstack([R, T])
    stereo = {"K0": K0, "dist0": d0, "K1": K1, "dist1": d1, "R": R, "T": T}
    errs = []
    for a, b in zip(p0[inliers], p1[inliers]):
        X = triangulate_point(tuple(a), tuple(b), K0, d0, K1, d1, P0, P1)
        errs.append(reprojection_error(X, tuple(a), tuple(b), stereo))
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    return {"R": R, "T": T, "E": E, "F": F, "P0": P0, "P1": P1,
            "inliers": int(inliers.sum()),
            "rms": float(np.sqrt(np.mean(np.square(errs)))) if errs else float("inf")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--baseline-mm", type=float, required=True,
                        help="Lens-to-lens distance measured with a ruler, in mm")
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    args = parser.parse_args()

    intr = []
    for idx in (args.cam0, args.cam1):
        f = args.calib_dir / f"cam{idx}_intrinsics.npz"
        if not f.exists():
            print(f"ERROR: {f} not found — run 02_calibrate_intrinsics.py --camera {idx} first.")
            sys.exit(1)
        d = np.load(f)
        intr.append((d["K"], d["dist"], tuple(d["img_size"].tolist())))
    (K0, d0, size0), (K1, d1, _) = intr

    caps = [cv2.VideoCapture(args.cam0), cv2.VideoCapture(args.cam1)]
    dets = [MotionDetector(min_area=args.min_area), MotionDetector(min_area=args.min_area)]
    s0: list = []
    s1: list = []
    prev = [None, None]
    frame_no = 0

    print(f"Lens-to-lens distance: {args.baseline_mm:.0f} mm")
    print("Keep both cameras still. Fly the target slowly through the shared view:")
    print("left/right, up/down and near/far. C = compute, X = clear, Q = quit.")

    while True:
        for c in caps:
            c.grab()
        frames = [c.retrieve()[1] for c in caps]
        if any(f is None for f in frames):
            print("ERROR: failed to read frames — is a camera unplugged?")
            break
        frame_no += 1
        pts = [dets[i](frames[i])[1] for i in range(2)]

        status = "learning background..." if frame_no <= WARMUP_FRAMES else "move the target slowly"
        if frame_no > WARMUP_FRAMES and pts[0] is not None and pts[1] is not None:
            widths = [frames[0].shape[1], frames[1].shape[1]]
            fast = any(prev[i] is not None and
                       math.dist(pts[i], prev[i]) > 0.025 * widths[i] for i in range(2))
            spaced = not s0 or math.dist(pts[0], s0[-1]) > 0.015 * widths[0]
            if fast:
                status = "TOO FAST - slow down"
            elif spaced:
                s0.append(pts[0])
                s1.append(pts[1])
                status = "sample added"
        prev = pts

        views = []
        coverage = []
        for i, (f, samples) in enumerate(zip(frames, (s0, s1))):
            v = f.copy()
            h, w = v.shape[:2]
            cells = {(min(int(x / w * GRID), GRID - 1), min(int(y / h * GRID), GRID - 1))
                     for x, y in samples}
            coverage.append(len(cells))
            for gx in range(GRID):
                for gy in range(GRID):
                    if (gx, gy) in cells:
                        x0, y0 = gx * w // GRID, gy * h // GRID
                        sub = v[y0:y0 + h // GRID, x0:x0 + w // GRID]
                        sub[:] = (sub * 0.75 + np.array([0, 80, 0]) * 0.25).astype(np.uint8)
            for x, y in samples:
                cv2.circle(v, (int(x), int(y)), 4, (0, 255, 255), -1)
            if pts[i] is not None:
                cv2.circle(v, (int(pts[i][0]), int(pts[i][1])), 20, (0, 230, 0), 3)
            sc = max(0.8, w / 1280)
            cv2.putText(v, f"CAM {(args.cam0, args.cam1)[i]}  coverage {len(cells)}/{GRID * GRID}",
                        (15, int(40 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * sc, (0, 230, 0), 2)
            views.append(v)

        ready = len(s0) >= 60
        banner_txt = (f"samples {len(s0)}   {status}   "
                      f"{'C = compute' if ready else 'need 60+ samples'}   X = clear   Q = quit")
        h = min(views[0].shape[0], views[1].shape[0], 480)
        resized = [cv2.resize(v, (int(v.shape[1] * h / v.shape[0]), h)) for v in views]
        banner = np.full((36, resized[0].shape[1] + resized[1].shape[1], 3), 25, np.uint8)
        color = (0, 140, 255) if "FAST" in status else (230, 230, 230)
        cv2.putText(banner, banner_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.imshow("Step 3b — Calibrate with a moving target", np.vstack([np.hstack(resized), banner]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            for c in caps:
                c.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord("x"):
            s0.clear()
            s1.clear()
        elif key == ord("r"):
            dets = [MotionDetector(min_area=args.min_area), MotionDetector(min_area=args.min_area)]
        elif key == ord("c") and ready:
            break

    for c in caps:
        c.release()
    cv2.destroyAllWindows()
    if len(s0) < 60:
        print("Not enough samples — exiting without saving.")
        sys.exit(1)

    p0, p1 = np.array(s0), np.array(s1)
    print(f"\nSolving camera geometry from {len(p0)} matched samples ...")
    try:
        r = solve_from_matches(p0, p1, K0, d0, K1, d1, args.baseline_mm)
    except (ValueError, cv2.error) as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print(f"  Inlier samples   : {r['inliers']} / {len(p0)}")
    print(f"  Reprojection RMS : {r['rms']:.2f} px")
    print(f"  Baseline         : {args.baseline_mm:.1f} mm  (your ruler measurement)")
    print_rig_geometry(r["R"], r["T"])
    if r["inliers"] < 0.6 * len(p0):
        print("WARNING: many samples rejected — hand or background motion was probably picked up.")
    if min(coverage) < 10:
        print("WARNING: low coverage — spread the target across more of both views.")
    if r["rms"] > 3:
        print("WARNING: RMS > 3 px — recollect, moving slower and covering near and far.")

    args.calib_dir.mkdir(parents=True, exist_ok=True)
    out = args.calib_dir / "stereo.npz"
    if out.exists():
        out.replace(args.calib_dir / "stereo_backup.npz")
        print("  (previous stereo.npz kept as stereo_backup.npz)")
    np.savez(out, K0=K0, dist0=d0, K1=K1, dist1=d1, R=r["R"], T=r["T"], E=r["E"], F=r["F"],
             P0=r["P0"], P1=r["P1"], baseline_mm=np.array([args.baseline_mm]),
             img_size=np.array(size0), method=np.array("motion"))
    np.savez(args.calib_dir / "motion_matches.npz", p0=p0, p1=p1)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
