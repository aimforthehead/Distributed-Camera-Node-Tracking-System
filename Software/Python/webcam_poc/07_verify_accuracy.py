#!/usr/bin/env python3
"""
07_verify_accuracy.py — Objective 3D accuracy test using the checkerboard as ground truth.

Hold the checkerboard where both cameras see it. Every inner corner is triangulated
in 3D with the same code the tracker uses, then compared with the board's known
geometry (square size, row/column spans, flatness). No hand-measured positions needed.

Optionally, type in a tape-measured distance from CAM 0 to the board centre to check
absolute range as well (this also catches a wrong --square value, which would scale
every distance).

Usage:
    python 07_verify_accuracy.py --cam0 0 --cam1 1 --board 9x6 --square 8

Hands-free by default: hold the board still and it measures automatically, then
move it to a new distance/angle. Report prints after --target measurements.
Use --manual to press SPACE yourself and type tape-measured distances.

Controls:
    SPACE — measure now
    Q     — quit and print the accuracy report

Output: accuracy_report.json and a printed summary for the proposal.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

from triangulate_utils import (load_calibration, triangulate_point, reprojection_error,
                               AutoCapture, beep, draw_auto_status, check_frame_size,
                               canonical_corners, refine_corners)


def parse_board(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    return int(parts[0]), int(parts[1])


def measure_board(corners0, corners1, stereo, board: tuple[int, int], square: float) -> dict:
    """Triangulate every corner and compare against the board's true geometry (mm)."""
    cols, rows = board
    c0 = corners0.reshape(-1, 2)
    c1 = corners1.reshape(-1, 2)
    args = (stereo["K0"], stereo["dist0"], stereo["K1"], stereo["dist1"], stereo["P0"], stereo["P1"])
    pts = np.array([triangulate_point(tuple(a), tuple(b), *args) for a, b in zip(c0, c1)])
    grid = pts.reshape(rows, cols, 3)

    spacing = np.concatenate([
        np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
    ])
    spacing_err = np.abs(spacing - square)

    row_span = np.linalg.norm(grid[:, -1] - grid[:, 0], axis=1)
    col_span = np.linalg.norm(grid[-1] - grid[0], axis=1)
    true_row, true_col = square * (cols - 1), square * (rows - 1)
    diag = np.linalg.norm(grid[-1, -1] - grid[0, 0])
    true_diag = square * np.hypot(cols - 1, rows - 1)

    centred = pts - pts.mean(axis=0)
    normal = np.linalg.svd(centred)[2][-1]
    planarity_rms = float(np.sqrt(np.mean((centred @ normal) ** 2)))

    reproj = [reprojection_error(p, tuple(a), tuple(b), stereo) for p, a, b in zip(pts, c0, c1)]

    return {
        "range_mm": float(np.linalg.norm(pts.mean(axis=0))),
        "spacing_mean_err_mm": float(spacing_err.mean()),
        "spacing_max_err_mm": float(spacing_err.max()),
        "spacing_mean_err_pct": float(spacing_err.mean() / square * 100),
        "span_err_mm": float(np.mean(np.abs(np.concatenate([row_span - true_row, col_span - true_col])))),
        "span_err_pct": float(np.mean(np.abs(np.concatenate([
            (row_span - true_row) / true_row, (col_span - true_col) / true_col]))) * 100),
        "diagonal_mm": float(diag),
        "diagonal_true_mm": float(true_diag),
        "diagonal_err_pct": float(abs(diag - true_diag) / true_diag * 100),
        "planarity_rms_mm": planarity_rms,
        "reproj_mean_px": float(np.mean(reproj)),
    }


def print_report(results: list[dict], square: float) -> None:
    print("\n" + "=" * 78)
    print("3D ACCURACY REPORT  (checkerboard ground truth, square = "
          f"{square:g} mm, {len(results)} placements)")
    print("=" * 78)
    print(f"{'#':>2} {'range':>8} {'spacing err':>13} {'span err':>10} {'diag err':>9} "
          f"{'flatness':>9} {'reproj':>7} {'tape err':>9}")
    for i, r in enumerate(results, 1):
        tape = f"{r['tape_err_pct']:.1f}%" if "tape_err_pct" in r else "-"
        print(f"{i:>2} {r['range_mm'] / 1000:7.2f}m {r['spacing_mean_err_mm']:8.2f} mm "
              f"{r['span_err_pct']:8.2f}% {r['diagonal_err_pct']:8.2f}% "
              f"{r['planarity_rms_mm']:7.2f}mm {r['reproj_mean_px']:5.1f}px {tape:>9}")
    rng = [r["range_mm"] / 1000 for r in results]
    span = np.mean([r["span_err_pct"] for r in results])
    spacing = np.mean([r["spacing_mean_err_mm"] for r in results])
    flat = np.mean([r["planarity_rms_mm"] for r in results])
    print("-" * 78)
    print(f"Range tested        : {min(rng):.2f} – {max(rng):.2f} m")
    print(f"Length error (span) : {span:.2f} % mean")
    print(f"Corner spacing error: {spacing:.2f} mm mean")
    print(f"Flatness (RMS)      : {flat:.2f} mm")
    tapes = [r for r in results if "tape_err_pct" in r]
    if tapes:
        print(f"Absolute range error: {np.mean([r['tape_err_pct'] for r in tapes]):.1f} % mean "
              f"({len(tapes)} tape-measured)")
    reproj = float(np.median([r["reproj_mean_px"] for r in results]))
    problems = []
    if reproj > 5:
        problems.append(f"reprojection {reproj:.0f} px (should be under ~3): the cameras moved or "
                        "refocused since step 03 — redo step 03 and don't touch them afterwards")
    if span > 8:
        problems.append(f"lengths off by {span:.0f}%: the board is not the size it was in step 03 "
                        "(phone zoom/rotation changed, or a different --square) — keep the phone image "
                        "untouched between step 03 and this check")
    if problems:
        print("\nRESULT: NOT VALID FOR THE PROPOSAL")
        for p in problems:
            print(f"  - {p}")
        return
    print("\nProposal line:")
    print(f"  \"At {min(rng):.1f}–{max(rng):.1f} m, the two-camera prototype measured known "
          f"lengths to within {span:.1f} % ({spacing:.1f} mm mean corner error).\"")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=0)
    parser.add_argument("--cam1", type=int, default=1)
    parser.add_argument("--board", type=parse_board, default=(9, 6), metavar="COLSxROWS")
    parser.add_argument("--square", type=float, required=True, help="Square size in mm")
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    parser.add_argument("--out", type=Path, default=Path("accuracy_report.json"))
    parser.add_argument("--target", type=int, default=6, help="Measurements before the report")
    parser.add_argument("--manual", action="store_true",
                        help="Press SPACE to measure and enter tape distances")
    args = parser.parse_args()

    stereo, _ = load_calibration(args.calib_dir, need_hsv=False)
    board = args.board
    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    check_frame_size(cap0, stereo["img_size"], f"camera {args.cam0}")
    if "img_size1" in stereo.files:
        check_frame_size(cap1, stereo["img_size1"], f"camera {args.cam1}")
    results: list[dict] = []

    print("Hold the board where BOTH cameras see it, keep it still, press SPACE.")
    print("Measure at several distances and angles. Q = quit and print the report.")
    auto = AutoCapture(min_change=0.08)

    while True:
        cap0.grab()
        cap1.grab()
        ok0, f0 = cap0.retrieve()
        ok1, f1 = cap1.retrieve()
        if not ok0 or not ok1:
            break
        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        found0, c0 = cv2.findChessboardCorners(g0, board, None, cv2.CALIB_CB_FAST_CHECK)
        found1, c1 = cv2.findChessboardCorners(g1, board, None, cv2.CALIB_CB_FAST_CHECK)

        d0, d1 = f0.copy(), f1.copy()
        if found0:
            cv2.drawChessboardCorners(d0, board, c0, True)
        if found1:
            cv2.drawChessboardCorners(d1, board, c1, True)
        both = found0 and found1
        shoot = False
        if not args.manual:
            shoot, state, progress = auto.update([c0, c1] if both else None,
                                                 [f0.shape[1], f1.shape[1]], time.time())
        msg = "BOTH SEE BOARD - hold still" if both else "board must be visible in both cameras"
        for d in (d0, d1):
            cv2.putText(d, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 220, 0) if both else (0, 0, 220), 2)
            cv2.putText(d, f"measurements: {len(results)}", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)
        if not args.manual:
            for d in (d0, d1):
                draw_auto_status(d, state, progress, len(results), args.target)
        h = min(d0.shape[0], d1.shape[0], 480)
        views = [cv2.resize(d, (int(d.shape[1] * h / d.shape[0]), h)) for d in (d0, d1)]
        cv2.imshow("Step 7 — Accuracy Check  |  SPACE=measure  Q=report", np.hstack(views))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if (shoot or key == ord(" ")) and both:
            if shoot:
                beep()
            c0 = canonical_corners(g0, refine_corners(g0, c0, board), board)
            c1 = canonical_corners(g1, refine_corners(g1, c1, board), board)
            r = measure_board(c0, c1, stereo, board, args.square)
            print(f"\n#{len(results) + 1}: range {r['range_mm'] / 1000:.2f} m | "
                  f"length error {r['span_err_pct']:.2f} % | spacing error "
                  f"{r['spacing_mean_err_mm']:.2f} mm | flatness {r['planarity_rms_mm']:.2f} mm | "
                  f"reprojection {r['reproj_mean_px']:.1f} px")
            tape = (input("  Tape-measured distance CAM 0 lens → board centre in mm (Enter to skip): ").strip()
                    if args.manual else "")
            if tape:
                try:
                    true_range = float(tape)
                    r["tape_range_mm"] = true_range
                    r["tape_err_pct"] = abs(r["range_mm"] - true_range) / true_range * 100
                    print(f"  Range: measured {r['range_mm']:.0f} mm vs tape {true_range:.0f} mm "
                          f"→ {r['tape_err_pct']:.1f} % error")
                except ValueError:
                    print("  Not a number — skipped.")
            r["timestamp"] = time.time()
            results.append(r)
            if not args.manual and len(results) >= args.target:
                break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    if not results:
        print("No measurements taken.")
        return
    print_report(results, args.square)
    with open(args.out, "w") as f:
        json.dump({"square_mm": args.square, "board": list(board), "measurements": results}, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
