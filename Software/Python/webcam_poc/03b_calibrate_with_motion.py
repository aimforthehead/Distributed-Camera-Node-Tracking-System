#!/usr/bin/env python3
"""
03b_calibrate_with_motion.py — Stereo calibration from a target moved through the view
(no checkerboard).

Alternative to step 03. Move the helicopter/drone (or any small object) to a spot both
cameras see, HOLD IT STILL for about a second (beep), then move to the next spot.
Each held spot gives one matched point pair; from ~40 spots the script solves how
camera 1 sits relative to camera 0 (essential-matrix method). The only physical
measurement needed is the lens-to-lens distance, measured with a ruler, which sets
the real-world scale.

Why hold still: two USB webcams don't take pictures at the same instant. While the
target moves, each camera sees it at a slightly different place and the pairs don't
match. A still target looks the same to both, whatever the timing.

This is how real rooftop nodes would be calibrated in the field: no calibration
pattern, just a drone hovering at a series of points in the shared view.

Still requires each camera's lens calibration from step 02.

Status: field-calibration prototype. With a few dozen spots of a hand-held target the
geometry is less certain than a checkerboard calibration; the script measures this
("geometry wobble") and refuses to save if it exceeds 1°. For the accuracy numbers,
use step 03 (checkerboard stereo).

Usage:
    python 03b_calibrate_with_motion.py --cam0 0 --cam1 1 --baseline-mm 400

For a good result:
    • hang the target on a thread or tape it to a thin stick; keep hands out of view
    • a small, compact target is more precise than a large one
    • spread the spots left/right, up/down AND near/far

Controls:
    C — compute now (needs ≥ 20 spots)
    X — clear spots      R — reset background      Q — quit
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np

from triangulate_utils import (MotionDetector, triangulate_point, reprojection_error,
                               print_rig_geometry, beep, check_frame_size)

WARMUP_FRAMES = 30
GRID = 4
STILL_FRAMES = 4          # consecutive frames the target must stay put in both views
STILL_TOL = 0.004         # max drift over those frames, as a fraction of image width
NEW_SPOT = 0.04           # a new spot must be this far (fraction of width) from all others
MIN_INLIER_RATIO = 0.5


def _residuals(x, u0, u1, p0, p1, K0, d0, K1, d1, baseline_mm):
    """Reprojection residuals (px) of triangulated points for pose x = [rvec, t]."""
    R = cv2.Rodrigues(x[:3])[0]
    T = (x[3:] / np.linalg.norm(x[3:]) * baseline_mm).reshape(3, 1)
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K1 @ np.hstack([R, T])
    Xh = cv2.triangulatePoints(P0, P1, u0, u1)
    X = (Xh[:3] / Xh[3]).T.reshape(-1, 1, 3)
    q0 = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K0, d0)[0].reshape(-1, 2)
    q1 = cv2.projectPoints(X, x[:3], T, K1, d1)[0].reshape(-1, 2)
    return np.concatenate([(q0 - p0).ravel(), (q1 - p1).ravel()])


def refine_pose(R, t, p0, p1, K0, d0, K1, d1, baseline_mm, iters=50):
    """Levenberg–Marquardt on rotation + baseline direction, minimising reprojection error."""
    u0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K0, d0, P=K0).reshape(-1, 2).T
    u1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2).T
    args = (u0, u1, p0, p1, K0, d0, K1, d1, baseline_mm)
    x = np.concatenate([cv2.Rodrigues(R)[0].ravel(), np.ravel(t) / np.linalg.norm(t)])
    r = _residuals(x, *args)
    cost, lam = r @ r, 1e-3
    for _ in range(iters):
        J = np.empty((len(r), 6))
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = 1e-6
            J[:, k] = (_residuals(x + dx, *args) - r) / 1e-6
        A, g = J.T @ J, J.T @ r
        improved = False
        while lam < 1e8:
            step = -np.linalg.solve(A + lam * (np.diag(np.diag(A)) + 1e-9 * np.eye(6)), g)
            xn = x + step
            xn[3:] /= np.linalg.norm(xn[3:])
            rn = _residuals(xn, *args)
            if rn @ rn < cost:
                x, r, cost, lam, improved = xn, rn, rn @ rn, max(lam / 10, 1e-9), True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-9:
            break
    return cv2.Rodrigues(x[:3])[0], x[3:].reshape(3, 1)


def solve_from_matches(p0: np.ndarray, p1: np.ndarray, K0, d0, K1, d1, baseline_mm: float) -> dict:
    """Relative pose of camera 1 from matched pixel points, scaled by the measured baseline."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    n0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K0, d0).reshape(-1, 2)
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K1, d1).reshape(-1, 2)
    f = (K0[0, 0] + K0[1, 1] + K1[0, 0] + K1[1, 1]) / 4
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.RANSAC,
                                   prob=0.999, threshold=3.0 / f)
    if E is None:
        raise ValueError("could not estimate camera geometry — collect more varied spots")
    _, R, t, mask = cv2.recoverPose(E[:3], n0, n1, np.eye(3), mask=mask)
    inliers = mask.ravel() > 0

    # Refine on the inliers, re-select inliers with the better model, refine again
    for _ in range(2):
        R, t = refine_pose(R, t, p0[inliers], p1[inliers], K0, d0, K1, d1, baseline_mm)
        u0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K0, d0, P=K0).reshape(-1, 2).T
        u1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2).T
        x = np.concatenate([cv2.Rodrigues(R)[0].ravel(), t.ravel()])
        res = _residuals(x, u0, u1, p0, p1, K0, d0, K1, d1, baseline_mm).reshape(2, -1, 2)
        err = np.linalg.norm(res, axis=2).max(axis=0)
        inliers = err < 4.0

    T = t.reshape(3, 1) * baseline_mm
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K1 @ np.hstack([R, T])
    tx = np.array([[0, -t[2, 0], t[1, 0]], [t[2, 0], 0, -t[0, 0]], [-t[1, 0], t[0, 0], 0]])
    E = tx @ R
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    rms = float(np.sqrt(np.mean(np.square(err[inliers])))) if inliers.any() else float("inf")
    return {"R": R, "T": T, "E": E, "F": F, "P0": P0, "P1": P1,
            "inliers": int(inliers.sum()), "mask": inliers, "rms": rms}


def pose_uncertainty(p0, p1, r, K0, d0, K1, d1, baseline_mm, n=12) -> float:
    """
    Re-solve on random resamples of the consistent spots; return the 90th-percentile
    rotation difference (degrees). Large values mean the spots don't pin the geometry
    down (rotation and sideways shift trade off against each other).
    """
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(r["mask"])
    devs = []
    for _ in range(n):
        pick = rng.choice(idx, len(idx), replace=True)
        try:
            rb = solve_from_matches(p0[pick], p1[pick], K0, d0, K1, d1, baseline_mm)
        except (ValueError, cv2.error):
            continue
        devs.append(np.degrees(np.linalg.norm(cv2.Rodrigues(rb["R"] @ r["R"].T)[0])))
    return float(np.percentile(devs, 90)) if devs else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--baseline-mm", type=float, required=True,
                        help="Lens-to-lens distance measured with a ruler, in mm")
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--target", type=int, default=40,
                        help="Auto-compute once this many spots and 10/16 coverage are reached")
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
    for cap, (_, _, size), idx in zip(caps, intr, (args.cam0, args.cam1)):
        check_frame_size(cap, size, f"camera {idx}")
    dets = [MotionDetector(min_area=args.min_area), MotionDetector(min_area=args.min_area)]
    s0: list = []
    s1: list = []
    recent = [deque(maxlen=STILL_FRAMES), deque(maxlen=STILL_FRAMES)]
    frame_no = 0
    flash_until = 0.0

    print(f"Lens-to-lens distance: {args.baseline_mm:.0f} mm")
    print("Keep both cameras still. Move the target to a spot, HOLD IT STILL until the beep,")
    print("then move to a new spot (left/right, up/down, near/far). Q = quit.")

    while True:
        for c in caps:
            c.grab()
        frames = [c.retrieve()[1] for c in caps]
        if any(f is None for f in frames):
            print("ERROR: failed to read frames — is a camera unplugged?")
            break
        frame_no += 1
        t = time.time()
        pts = [dets[i](frames[i])[1] for i in range(2)]
        widths = [frames[0].shape[1], frames[1].shape[1]]

        if frame_no <= WARMUP_FRAMES:
            status = "LEARNING BACKGROUND - keep everything still"
        elif pts[0] is None or pts[1] is None:
            status = "TARGET NOT SEEN BY BOTH CAMERAS"
            for r in recent:
                r.clear()
        else:
            for i in range(2):
                recent[i].append(pts[i])
            full = all(len(r) == STILL_FRAMES for r in recent)
            still = full and all(
                max(math.dist(p, np.mean(recent[i], axis=0)) for p in recent[i]) < STILL_TOL * widths[i]
                for i in range(2))
            spot0 = np.mean(recent[0], axis=0)
            new = not s0 or min(math.dist(spot0, s) for s in s0) > NEW_SPOT * widths[0]
            if t < flash_until:
                status = f"SPOT {len(s0)} SAVED - move to a new spot"
            elif not new:
                status = "MOVE TO A NEW SPOT"
            elif still:
                s0.append(tuple(spot0))
                s1.append(tuple(np.mean(recent[1], axis=0)))
                beep()
                print(f"  spot {len(s0)}")
                flash_until = t + 1.0
                status = f"SPOT {len(s0)} SAVED - move to a new spot"
                for r in recent:
                    r.clear()
            else:
                status = "HOLD STILL..."

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
                cv2.circle(v, (int(x), int(y)), 6, (0, 255, 255), -1)
            if pts[i] is not None:
                cv2.circle(v, (int(pts[i][0]), int(pts[i][1])), 22, (0, 230, 0), 3)
            sc = max(0.8, w / 1280)
            cv2.putText(v, f"CAM {(args.cam0, args.cam1)[i]}  coverage {len(cells)}/{GRID * GRID}",
                        (15, int(40 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * sc, (0, 230, 0), 2)
            views.append(v)

        if len(s0) >= args.target and min(coverage) >= 10:
            print("Enough spots and coverage — computing automatically.")
            break

        h = min(views[0].shape[0], views[1].shape[0], 480)
        resized = [cv2.resize(v, (int(v.shape[1] * h / v.shape[0]), h)) for v in views]
        banner = np.full((64, resized[0].shape[1] + resized[1].shape[1], 3), 25, np.uint8)
        color = {"HOLD": (0, 220, 255), "SPOT": (0, 230, 0), "MOVE": (0, 160, 255)}.get(
            status.split()[0], (200, 200, 255))
        cv2.putText(banner, status, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(banner, f"spots {len(s0)}/{args.target}", (banner.shape[1] - 230, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.imshow("Step 3b — Calibrate with a target  |  C=compute  X=clear  Q=quit",
                   np.vstack([np.hstack(resized), banner]))

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
        elif key == ord("c") and len(s0) >= 20:
            break

    for c in caps:
        c.release()
    cv2.destroyAllWindows()
    if len(s0) < 20:
        print("Not enough spots — exiting without saving.")
        sys.exit(1)

    p0, p1 = np.array(s0), np.array(s1)
    args.calib_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.calib_dir / "motion_matches.npz", p0=p0, p1=p1)
    print(f"\nSolving camera geometry from {len(p0)} spots ...")
    try:
        r = solve_from_matches(p0, p1, K0, d0, K1, d1, args.baseline_mm)
    except (ValueError, cv2.error) as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    ratio = r["inliers"] / len(p0)
    wobble = pose_uncertainty(p0, p1, r, K0, d0, K1, d1, args.baseline_mm)
    print(f"  Consistent spots : {r['inliers']} / {len(p0)}  ({ratio:.0%})")
    print(f"  Geometry wobble  : ±{wobble:.2f}°  (how much the answer shifts on resampling)")
    print(f"  Reprojection RMS : {r['rms']:.2f} px")
    print(f"  Baseline         : {args.baseline_mm:.1f} mm  (your ruler measurement)")
    print_rig_geometry(r["R"], r["T"])

    if ratio < MIN_INLIER_RATIO or r["inliers"] < 15:
        print("\nRESULT: REJECTED — too few spots agree, so the geometry can't be trusted. NOT saved.")
        print("  Usually the cameras locked onto different things (hand, arm, shadow) or the")
        print("  target swung during a capture. Keep hands out of view and hold each spot still.")
        sys.exit(1)
    if wobble > 1.0:
        print("\nRESULT: REJECTED — the spots don't pin the camera geometry down precisely")
        print(f"  (answer shifts by up to {wobble:.1f}° on resampling). NOT saved.")
        print("  Use more spots spread wider and nearer/farther, a smaller target, or use the")
        print("  checkerboard stereo calibration (step 03) for measured accuracy.")
        sys.exit(1)
    if min(coverage) < 10:
        print("WARNING: low coverage — spread the spots across more of both views.")

    out = args.calib_dir / "stereo.npz"
    if out.exists():
        out.replace(args.calib_dir / "stereo_backup.npz")
        print("  (previous stereo.npz kept as stereo_backup.npz)")
    np.savez(out, K0=K0, dist0=d0, K1=K1, dist1=d1, R=r["R"], T=r["T"], E=r["E"], F=r["F"],
             P0=r["P0"], P1=r["P1"], baseline_mm=np.array([args.baseline_mm]),
             img_size=np.array(size0), img_size1=np.array(intr[1][2]), method=np.array("motion"))
    print(f"RESULT: {'GOOD' if r['rms'] < 2 and ratio > 0.8 else 'OK'} — saved → {out}")


if __name__ == "__main__":
    main()
