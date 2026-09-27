#!/usr/bin/env python3
"""
02_calibrate_intrinsics.py — Calibrate one camera's intrinsic parameters (K, distortion).

Run this separately for each camera.  Move the checkerboard to fill the frame
and tilt it to various angles — flat-on views give poor calibration.

Usage:
    python 02_calibrate_intrinsics.py --camera 0 --screen      # board on the laptop screen (easiest)
    python 02_calibrate_intrinsics.py --camera 1 --board 9x6 --square 9.6   # hand-held phone board

--screen draws the checkerboard full-screen on this computer, with a map of what the
camera sees and where to aim next. Set the webcam down 50–80 cm from the screen,
move it after each beep. Steadier than a hand-held phone, so the result is sharper.

    --board   Inner corners as COLSxROWS (count black-black intersections, not squares)
              A standard printed 10×7 grid has 9×6 inner corners.
    --square  Square size in mm. Only scales the board; the lens result does not depend on it.
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
    parser.add_argument("--square", type=float, default=25.0,
                        help="Square size in mm (does not affect the lens result)")
    parser.add_argument("--screen", action="store_true",
                        help="Show the checkerboard full-screen on this computer and move the camera")
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

    if args.screen:
        imgpoints, img_size = run_screen_mode(cap, args.board, args.target, subpix_criteria,
                                              args.camera)
        objpoints = [obj_tmpl.copy() for _ in imgpoints]

    while not args.screen:
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
    rms, K, dist, dropped = calibrate(objpoints, imgpoints, img_size)
    if dropped:
        print(f"  Dropped {dropped} blurry/shaky captures and recalibrated without them")
    w, h = img_size
    hfov = 2 * np.degrees(np.arctan(w / 2 / K[0, 0]))
    print(f"  RMS reprojection error : {rms:.4f} px")
    print(f"  Focal length           : fx={K[0,0]:.1f}  fy={K[1,1]:.1f} px  "
          f"(horizontal field of view {hfov:.0f}°)")
    print(f"  Principal point        : ({K[0,2]:.1f}, {K[1,2]:.1f})  (image centre {w/2:.0f}, {h/2:.0f})")
    print(f"  Distortion k1,k2       : {dist.flatten()[:2]}")

    off_centre = max(abs(K[0, 2] - w / 2) / w, abs(K[1, 2] - h / 2) / h)
    if rms > 2.0 or off_centre > 0.15 or abs(dist.flatten()[1]) > 3:
        print("RESULT: REDO — hold the board sharper/steadier (for fixed-focus cameras stay ≥ 40 cm away), "
              "cover all four corners of the view and tilt more.")
    elif rms > 1.0:
        print("RESULT: OK for the prototype (RMS under 2 px).")
    else:
        print("RESULT: GOOD.")

    np.savez(out_file, K=K, dist=dist, img_size=np.array(img_size), rms=np.array(rms))
    print(f"Saved → {out_file}")


CANVAS_W, CANVAS_H, PANEL_X = 1600, 1000, 1180
CELL_ORDER = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (0, 1), (2, 1), (1, 2), (1, 1)]
VIEWPOINTS = ["straight in front of the screen", "from the LEFT side, at an angle",
              "from the RIGHT side, at an angle", "from ABOVE, looking down at the screen",
              "from BELOW, looking up at the screen"]


def screen_board(board: tuple[int, int]) -> np.ndarray:
    """White canvas with the checkerboard on the left and room for the guide panel on the right."""
    cols, rows = board[0] + 1, board[1] + 1
    sq = min((PANEL_X - 20) // (cols + 2), (CANVAS_H - 20) // (rows + 2))
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, np.uint8)
    x0 = 10 + ((PANEL_X - 20) - sq * (cols + 2)) // 2
    y0 = (CANVAS_H - sq * (rows + 2)) // 2
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = y0 + (r + 1) * sq, x0 + (c + 1) * sq
                canvas[y:y + sq, x:x + sq] = 0
    cv2.line(canvas, (PANEL_X, 0), (PANEL_X, CANVAS_H), (200, 200, 200), 2)
    return canvas


def aim_hint(cell: tuple[int, int]) -> str:
    gx, gy = cell
    moves = []
    if gx == 0:
        moves.append("turn camera RIGHT")
    elif gx == 2:
        moves.append("turn camera LEFT")
    if gy == 0:
        moves.append("tilt it DOWN")
    elif gy == 2:
        moves.append("tilt it UP")
    return " + ".join(moves) if moves else "point camera straight at the board"


def draw_guide(canvas, frame_shape, corners, board, hits, target_cell, state, progress,
               n, target, viewpoint) -> np.ndarray:
    img = canvas.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    px, pw = PANEL_X + 30, CANVAS_W - PANEL_X - 60
    h, w = frame_shape[:2]
    ph, top = int(pw * h / w), 100
    cv2.putText(img, "WHAT THE CAMERA SEES", (px, 70), font, 0.75, (60, 60, 60), 2, cv2.LINE_AA)
    for gy in range(3):
        for gx in range(3):
            x1, y1 = px + gx * pw // 3, top + gy * ph // 3
            x2, y2 = px + (gx + 1) * pw // 3, top + (gy + 1) * ph // 3
            if hits[gy][gx]:
                cv2.rectangle(img, (x1, y1), (x2, y2), (190, 235, 190), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (210, 210, 210), 1)
    tx, ty = target_cell
    cv2.rectangle(img, (px + tx * pw // 3 + 3, top + ty * ph // 3 + 3),
                  (px + (tx + 1) * pw // 3 - 3, top + (ty + 1) * ph // 3 - 3), (0, 200, 255), 5)
    cv2.rectangle(img, (px, top), (px + pw, top + ph), (40, 40, 40), 2)
    if corners is not None:
        pts = corners.reshape(-1, 2)
        quad = pts[[0, board[0] - 1, len(pts) - 1, len(pts) - board[0]]]
        quad = np.column_stack([px + quad[:, 0] / w * pw, top + quad[:, 1] / h * ph]).astype(np.int32)
        cv2.polylines(img, [quad], True, (0, 170, 0), 4, cv2.LINE_AA)

    text, color = {
        "none": ("BOARD NOT IN VIEW", (0, 0, 220)),
        "move": ("MOVE THE CAMERA", (0, 140, 255)),
        "hold": ("HOLD STILL...", (0, 160, 200)),
        "captured": ("CAPTURED!", (0, 160, 0)),
    }[state]
    y = top + ph + 70
    cv2.putText(img, text, (px, y), font, 1.2, color, 3, cv2.LINE_AA)
    bar = int(pw * (progress if state == "hold" else (1.0 if state == "captured" else 0)))
    cv2.rectangle(img, (px, y + 18), (px + bar, y + 30), color, -1)
    cv2.putText(img, f"{n} / {target} captured", (px, y + 75), font, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
    lines = ["Move the camera until the green", "outline sits in the yellow box:",
             f"  {aim_hint(target_cell)}", "", "View the screen:", f"  {viewpoint}", "",
             "Set the camera down, wait for", "the beep, then move it again.", "", "Q = quit"]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (px, y + 130 + i * 34), font, 0.62, (50, 50, 50), 1, cv2.LINE_AA)
    return img


def run_screen_mode(cap, board, target, subpix, cam_idx):
    """Full-screen checkerboard with a live guide; the user moves the camera, not the board."""
    win = f"Lens calibration - camera {cam_idx}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas = screen_board(board)
    auto = AutoCapture(min_change=0.04)
    hits = [[0] * 3 for _ in range(3)]
    imgpoints: list[np.ndarray] = []
    img_size = None
    print("Board is on screen. Set the webcam 50-80 cm away facing it; move it after each beep.")
    while len(imgpoints) < target:
        ok, frame = cap.read()
        if not ok:
            break
        img_size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board, None)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix)
        shoot, state, progress = auto.update([corners] if found else None, [frame.shape[1]], time.time())
        if shoot:
            imgpoints.append(corners)
            c = corners.reshape(-1, 2).mean(axis=0)
            hits[min(int(c[1] / frame.shape[0] * 3), 2)][min(int(c[0] / frame.shape[1] * 3), 2)] += 1
            beep()
            print(f"  Captured frame {len(imgpoints)}/{target}")
        target_cell = min(CELL_ORDER, key=lambda cell: (hits[cell[1]][cell[0]], CELL_ORDER.index(cell)))
        viewpoint = VIEWPOINTS[len(imgpoints) % len(VIEWPOINTS)]
        cv2.imshow(win, draw_guide(canvas, frame.shape, corners if found else None, board, hits,
                                   target_cell, state, progress, len(imgpoints), target, viewpoint))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        if key == ord("c") and len(imgpoints) >= 8:
            break
    cv2.destroyAllWindows()
    return imgpoints, img_size


def calibrate(objpoints, imgpoints, img_size):
    """
    Webcam-constrained calibration: square pixels, no tangential distortion, no k3.
    Fewer free parameters keeps blurry hand-held captures from producing absurd lens
    models. Captures whose error is far above the rest are dropped and the fit redone.
    """
    w, h = img_size
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    K0 = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)

    def run(obj, img):
        rms, K, dist, _r, _t, _si, _se, per_view = cv2.calibrateCameraExtended(
            obj, img, img_size, K0.copy(), np.zeros(5), flags=flags)
        return rms, K, dist, per_view.ravel()

    rms, K, dist, per_view = run(objpoints, imgpoints)
    keep = per_view <= max(2.0 * np.median(per_view), 1.0)
    if keep.sum() >= 8 and not keep.all():
        obj = [o for o, k in zip(objpoints, keep) if k]
        img = [i for i, k in zip(imgpoints, keep) if k]
        rms, K, dist, _ = run(obj, img)
    return rms, K, dist, int((~keep).sum()) if keep.sum() >= 8 else 0


if __name__ == "__main__":
    main()
