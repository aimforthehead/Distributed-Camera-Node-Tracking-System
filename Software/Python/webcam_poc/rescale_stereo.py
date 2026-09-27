#!/usr/bin/env python3
"""
rescale_stereo.py — Correct the real-world scale of calibration/stereo.npz without recapturing.

The stereo calibration's angles don't depend on the checkerboard square size, but every
distance scales with it. If the square size used was wrong, or you trust a ruler
measurement of the lens-to-lens distance more, fix the scale here.

Usage (pick one):
    python rescale_stereo.py --used-square 9.6 --true-square 7.94
    python rescale_stereo.py --baseline-mm 400
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--used-square", type=float, help="Square size given to step 03 (mm)")
    parser.add_argument("--true-square", type=float, help="Correct square size (mm)")
    parser.add_argument("--baseline-mm", type=float, help="Ruler lens-to-lens distance (mm)")
    parser.add_argument("--calib_dir", type=Path, default=Path("calibration"))
    args = parser.parse_args()

    path = args.calib_dir / "stereo.npz"
    if not path.exists():
        print(f"ERROR: {path} not found — run step 03 first.")
        sys.exit(1)
    data = dict(np.load(path))
    old = float(np.linalg.norm(data["T"]))

    if args.baseline_mm:
        factor = args.baseline_mm / old
    elif args.used_square and args.true_square:
        factor = args.true_square / args.used_square
    else:
        print("Give either --baseline-mm, or both --used-square and --true-square.")
        sys.exit(1)

    data["T"] = data["T"] * factor
    data["P1"] = data["K1"] @ np.hstack([data["R"], data["T"]])
    data["baseline_mm"] = np.array([old * factor])
    path.replace(args.calib_dir / "stereo_before_rescale.npz")
    np.savez(path, **data)
    print(f"Baseline {old:.1f} mm → {old * factor:.1f} mm (×{factor:.4f}). Angles unchanged.")
    print(f"Saved → {path}  (previous file kept as stereo_before_rescale.npz)")


if __name__ == "__main__":
    main()
