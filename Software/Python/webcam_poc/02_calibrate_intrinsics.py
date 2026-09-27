#!/usr/bin/env python3
"""
02_calibrate_intrinsics.py — Calibrate one camera's intrinsic parameters (K, distortion).

Run this separately for each camera.  Move the checkerboard to fill the frame
and tilt it to various angles — flat-on views give poor calibration.

Usage:
    python 02_calibrate_intrinsics.py --camera 0 --board 9x6 --square 25
    python 02_calibrate_intrinsics.py --camera 1 --board 9x6 --square 25

    --board   Inner corners as COLSxROWS (count black-black intersections, not squares)
              A standard printed 10×7 grid has 9×6 inner corners.
    --square  Physical square side length in mm (sets the unit of the 3D result)
    --output  Directory to save calibration files (default: calibration/)

Hands-free by default: hold the board still for about a second and it captures
automatically (beep + green flash), then move or tilt it to a new position.
It calibrates by itself after --target captures (default 20).

Controls:
    SPACE — capture now (manual)
    C     — compute calibration early (need ≥ 8 frames)
    Q     — quit without saving

Output: calibration/camN_intrinsics.npz  →  K, dist, img_size, rms
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from triangulate_utils import AutoCapture, beep, draw_auto_status


def parse_board(s: str) -> tuple[int, int]:
    parts = s.lower().split('x')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected COLSxROWS, got '{s}'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--board", type=parse_board, default=(9, 6),
                        metavar="COLSxROWS")
    parser.add_argument("--square", type=float, required=True,
                        help="Square size in mm")
    parser.add_argument("--output", type=Path, default=Path("calibration"))
    parser.add_argument("--target", type=int, default=20, help="Captures before auto-calibrating")
    parser.add_argument("--manual", action="store_true", help="Disable auto-capture")
    args = parser.parse_args()

    board_w, board_h = args.board
    sq = args.square
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cam{args.camera}_intrinsics.npz"

    # Real-world 3D coords of corners in the Z=0 plane (mm)
    obj_tmpl = np.zeros((board_w * board_h, 3), np.float32)
    obj_tmpl[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    obj_tmpl *= sq

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}")
        sys.exit(1)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size: tuple[int, int] | None = None

    print(f"Camera {args.camera}  |  Board {board_w}×{board_h}  |  Square {sq} mm")
    print("Hold the board still to auto-capture; move/tilt between captures.")
    print("Controls: SPACE = capture  |  C = calibrate  |  Q = quit")
    auto = AutoCapture()
    print("Tip: use ≥ 15 frames with varied angles and positions across the frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if img_size is None:
            img_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

        display = frame.copy()
        if found:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            cv2.drawChessboardCorners(display, (board_w, board_h), corners_sub, True)

        if not args.manual:
            shoot, state, progress = auto.update([corners_sub] if found else None,
                                                 [frame.shape[1]], time.time())
            if shoot:
                objpoints.append(obj_tmpl.copy())
                imgpoints.append(corners_sub)
                beep()
                print(f"  Captured frame {len(objpoints)}/{args.target}")
                if len(objpoints) >= args.target:
                    break
            draw_auto_status(display, state, progress, len(objpoints), args.target)
        else:
            msg = (f"FOUND — {len(objpoints)} captured  |  SPACE to grab" if found
                   else f"Board not found — {len(objpoints)} captured")
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 220, 0) if found else (0, 0, 220), 2)

        cv2.imshow(f"Step 2 — Intrinsic Calibration  Cam {args.camera}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord(' ') and found:
            objpoints.append(obj_tmpl.copy())
            imgpoints.append(corners_sub)
            print(f"  Captured frame {len(objpoints)}")
        elif key == ord('c'):
            if len(objpoints) < 8:
                print(f"Need at least 8 frames (have {len(objpoints)})")
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 8:
        print("Not enough frames — exiting without saving.")
        sys.exit(1)

    print(f"\nCalibrating with {len(objpoints)} frames ...")
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    print(f"  RMS reprojection error : {rms:.4f} px")
    print(f"  Focal length           : fx={K[0,0]:.1f}  fy={K[1,1]:.1f} px")
    print(f"  Principal point        : ({K[0,2]:.1f}, {K[1,2]:.1f})")
    print(f"  Distortion k1,k2,p1,p2 : {dist.flatten()[:4]}")

    if rms > 1.0:
        print("WARNING: RMS > 1.0 px — consider recapturing with more varied angles.")

    np.savez(out_file, K=K, dist=dist, img_size=np.array(img_size), rms=np.array(rms))
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
