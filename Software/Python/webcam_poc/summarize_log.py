#!/usr/bin/env python3
"""
summarize_log.py — Summarise a tracking log from 05_triangulate.py and plot the flight.

Usage:
    python summarize_log.py                       # tracking_log.csv
    python summarize_log.py helicopter_run1.csv

Prints tracking rate, range covered, speeds, match quality and glitches, and saves
<log name>_summary.png (top-down path, range over time, match error over time).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

JUMP_MM = 150  # a frame-to-frame 3D jump larger than this is counted as a glitch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", type=Path, default=Path("tracking_log.csv"))
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.log)))
    if not rows:
        print("Log is empty.")
        return
    t = np.array([float(r["timestamp_s"]) for r in rows])
    t -= t[0]
    det = np.array([r["detected"] == "1" for r in rows])
    reproj = np.array([float(r["reproj_px"]) if r.get("reproj_px") else np.nan for r in rows])

    def col(name):
        return np.array([float(r[name]) if r[name] else np.nan for r in rows])

    x, y, z, rng = col("x_mm"), col("y_mm"), col("z_mm"), col("range_mm")
    speed, closing = col("speed_mps"), col("closing_mps")

    dur = t[-1] if len(t) > 1 else 0.0
    n, nd = len(rows), int(det.sum())
    segments, run = [], 0
    for d in det:
        if d:
            run += 1
        elif run:
            segments.append(run)
            run = 0
    if run:
        segments.append(run)
    fps = (n - 1) / dur if dur > 0 else 0.0
    rejected = int(np.sum(~det & ~np.isnan(reproj)))

    print(f"Log: {args.log}")
    print(f"  Duration          : {dur:.1f} s, {n} frames ({fps:.1f} fps)")
    print(f"  Target tracked    : {nd} frames ({100 * nd / n:.0f}%), in {len(segments)} tracks, "
          f"longest {max(segments, default=0) / max(fps, 1e-9):.1f} s")
    print(f"  Rejected (no match): {rejected} frames where both cameras saw something but it didn't match")
    if nd:
        p = np.column_stack([x, y, z])[det]
        jumps = np.linalg.norm(np.diff(p, axis=0), axis=1)
        print(f"  Range             : {np.nanmin(rng[det]) / 1000:.2f} – {np.nanmax(rng[det]) / 1000:.2f} m "
              f"(mean {np.nanmean(rng[det]) / 1000:.2f} m)")
        print(f"  X left/right      : {np.nanmin(x[det]) / 1000:+.2f} to {np.nanmax(x[det]) / 1000:+.2f} m")
        print(f"  Height (−Y)       : {np.nanmin(-y[det]) / 1000:+.2f} to {np.nanmax(-y[det]) / 1000:+.2f} m")
        print(f"  Top speed         : {np.nanmax(speed[det]):.2f} m/s")
        print(f"  Max approach speed: {np.nanmax(closing[det]):.2f} m/s "
              f"({int(np.sum(closing[det] > 0.1))} frames flagged APPROACHING)")
        print(f"  Match error       : median {np.nanmedian(reproj[det]):.1f} px, "
              f"95th percentile {np.nanpercentile(reproj[det], 95):.1f} px")
        print(f"  Glitches (>{JUMP_MM} mm jump between frames): {int(np.sum(jumps > JUMP_MM))}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping plot)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(x[det] / 1000, z[det] / 1000, ".-", ms=3, lw=0.8)
    ax[0].plot([0], [0], "k^", ms=10, label="camera 0")
    ax[0].set(xlabel="X left/right (m)", ylabel="Z forward (m)", title="Top-down flight path")
    ax[0].axis("equal")
    ax[0].legend()
    ax[1].plot(t, rng / 1000, lw=1)
    ax[1].set(xlabel="time (s)", ylabel="range (m)", title="Range over time")
    ax[2].plot(t, reproj, lw=1)
    ax[2].set(xlabel="time (s)", ylabel="px", title="Match error (lower is better)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = args.log.with_name(args.log.stem + "_summary.png")
    fig.savefig(out, dpi=120)
    print(f"  Plot saved        : {out}")


if __name__ == "__main__":
    main()
