#!/usr/bin/env python3
"""
03_calibrate_extrinsics.py — Stereo calibration: measure camera 1's pose relative to camera 0.

Both cameras must see the checkerboard AT THE SAME TIME for every captured pair.
Aim both cameras at the same volume of space so they have an overlapping field of view,
then move the checkerboard through that overlap.

Requires: calibration/cam0_intrinsics.npz and calibration/cam1_intrinsics.npz (from step 02).

Usage:
    python 03_calibrate_extrinsics.py --board 9x6 --square 25
    python 03_calibrate_extrinsics.py --cam0 0 --cam1 1 --board 9x6 --square 25

Controls:
    SPACE — capture stereo pair (only works when BOTH cameras see the board)
    C     — compute (need ≥ 10 pairs; ≥ 15 recommended)
    Q     — quit

Output: calibration/stereo.npz
    K0, dist0, K1, dist1 — intrinsics (copied from step 02)
    R, T                  — rotation & translation of cam1 relative to cam0
    P0, P1                — 3×4 projection matrices for triangulation
    baseline_mm           — measured distance between camera optical centres
    img_size              — (width, height)
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np


def parse_board(s: str) -> tuple[int, int]:
    parts = s.lower().split('x')
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--board", type=parse_board, default=(9, 6), metavar="COLSxROWS")
    parser.add_argument("--square", type=float, required=True, help="Square size in mm")
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    args = parser.parse_args()

    board_w, board_h = args.board
    sq = args.square
    calib_dir = args.calib_dir

    for cam in [0, 1]:
        f = calib_dir / f"cam{cam}_intrinsics.npz"
        if not f.exists():
            print(f"ERROR: {f} not found — run 02_calibrate_intrinsics.py --camera {cam} first.")
            sys.exit(1)

    d0 = np.load(calib_dir / "cam0_intrinsics.npz")
    d1 = np.load(calib_dir / "cam1_intrinsics.npz")
    K0, dist0 = d0["K"], d0["dist"]
    K1, dist1 = d1["K"], d1["dist"]
    img_size = tuple(d0["img_size"].tolist())  # (width, height)

    obj_tmpl = np.zeros((board_w * board_h, 3), np.float32)
    obj_tmpl[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    obj_tmpl *= sq

    objpoints: list[np.ndarray] = []
    imgpoints0: list[np.ndarray] = []
    imgpoints1: list[np.ndarray] = []

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"Board {board_w}×{board_h}  |  Square {sq} mm")
    print("Both cameras must see the board SIMULTANEOUSLY.")
    print("Controls: SPACE = capture pair  |  C = compute  |  Q = quit")

    while True:
        cap0.grab()
        cap1.grab()
        ok0, frame0 = cap0.retrieve()
        ok1, frame1 = cap1.retrieve()
        if not ok0 or not ok1:
            break

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        found0, corners0 = cv2.findChessboardCorners(gray0, (board_w, board_h), None)
        found1, corners1 = cv2.findChessboardCorners(gray1, (board_w, board_h), None)

        disp0, disp1 = frame0.copy(), frame1.copy()
        c0r = c1r = None

        if found0:
            c0r = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), subpix)
            cv2.drawChessboardCorners(disp0, (board_w, board_h), c0r, True)
        if found1:
            c1r = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), subpix)
            cv2.drawChessboardCorners(disp1, (board_w, board_h), c1r, True)

        both = found0 and found1
        for disp, found, label in [(disp0, found0, "Cam0"), (disp1, found1, "Cam1")]:
            status = "FOUND" if found else "not found"
            color = (0, 220, 0) if found else (0, 0, 220)
            cv2.putText(disp, f"{label}: {status}  pairs: {len(objpoints)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if both:
                cv2.putText(disp, "SPACE to capture", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2)

        h = min(disp0.shape[0], disp1.shape[0], 480)
        def rh(f: np.ndarray) -> np.ndarray:
            s = h / f.shape[0]
            return cv2.resize(f, (int(f.shape[1] * s), h))

        cv2.imshow("Step 3 — Stereo Calibration  |  SPACE=capture  C=compute  Q=quit",
                   np.hstack([rh(disp0), rh(disp1)]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap0.release(); cap1.release(); cv2.destroyAllWindows(); sys.exit(0)
        elif key == ord(' ') and both:
            objpoints.append(obj_tmpl.copy())
            imgpoints0.append(c0r)
            imgpoints1.append(c1r)
            print(f"  Captured pair {len(objpoints)}")
        elif key == ord('c'):
            if len(objpoints) < 10:
                print(f"Need at least 10 pairs (have {len(objpoints)})")
            else:
                break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        print("Not enough pairs — exiting without saving.")
        sys.exit(1)

    print(f"\nRunning stereo calibration with {len(objpoints)} pairs ...")
    # CALIB_FIX_INTRINSIC: trust the per-camera calibrations from step 02,
    # only solve for R and T between the two cameras.
    rms, K0_o, dist0_o, K1_o, dist1_o, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints0, imgpoints1,
        K0, dist0, K1, dist1,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    baseline = float(np.linalg.norm(T))
    print(f"  Stereo RMS error : {rms:.4f} px")
    print(f"  Baseline         : {baseline:.1f} mm  ({baseline/10:.1f} cm)")

    if rms > 1.5:
        print("WARNING: RMS > 1.5 px — consider recapturing; make sure the board fills the frame.")

    # Projection matrices:
    #   P0 = K0 · [I | 0]   (cam0 is world origin)
    #   P1 = K1 · [R | T]   (cam1 expressed in cam0 frame)
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K1 @ np.hstack([R, T])

    out = calib_dir / "stereo.npz"
    np.savez(out,
             K0=K0, dist0=dist0, K1=K1, dist1=dist1,
             R=R, T=T, E=E, F=F,
             P0=P0, P1=P1,
             baseline_mm=np.array([baseline]),
             img_size=np.array(img_size))
    print(f"Saved → {out}")
    print("\nCoordinate system from now on:")
    print("  Origin = camera 0 optical centre")
    print("  +X = cam0 right, +Y = cam0 down, +Z = cam0 forward (into scene)")
    print(f"  Units = mm  (checkerboard square = {sq} mm)")


if __name__ == "__main__":
    main()
