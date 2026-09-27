#!/usr/bin/env python3
"""
01_check_cameras.py — Verify both webcams are accessible and show live side-by-side feeds.

Usage:
    python 01_check_cameras.py                  # show ALL cameras labelled by index
    python 01_check_cameras.py --cam0 0 --cam1 2

Controls:
    Q — quit
    S — save snapshot of both frames
"""
from __future__ import annotations
import argparse
import sys
import cv2
import numpy as np


def find_cameras(max_index: int = 8) -> list[int]:
    """Return list of working camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def show_all(indices: list[int]) -> None:
    """Show every camera in a grid, labelled with its index, to tell them apart."""
    caps = {i: cv2.VideoCapture(i) for i in indices}
    for i, cap in caps.items():
        print(f"  index {i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("\nCover each lens with a finger to see which index it is. Q = quit")
    tw, th = 480, 270
    cols = 2 if len(caps) <= 4 else 3
    while True:
        tiles = []
        for i, cap in caps.items():
            ok, f = cap.read()
            tile = np.zeros((th, tw, 3), np.uint8) if not ok else cv2.resize(f, (tw, th))
            cv2.rectangle(tile, (0, 0), (tw, 44), (0, 0, 0), -1)
            label = f"INDEX {i}  {f.shape[1]}x{f.shape[0]}" if ok else f"INDEX {i}  no frame"
            cv2.putText(tile, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 0), 2)
            tiles.append(tile)
        while len(tiles) % cols:
            tiles.append(np.zeros((th, tw, 3), np.uint8))
        rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
        cv2.imshow("All cameras  |  Q = quit", np.vstack(rows))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=None)
    parser.add_argument("--cam1", type=int, default=None)
    args = parser.parse_args()

    if args.cam0 is None and args.cam1 is None:
        print("Scanning for cameras (indices 0–7)...")
        found = find_cameras()
        print(f"Found cameras at indices: {found}")
        if not found:
            print("ERROR: no cameras found")
            sys.exit(1)
        show_all(found)
        return

    if args.cam0 is None or args.cam1 is None:
        print("Scanning for cameras (indices 0–7)...")
        found = find_cameras()
        print(f"Found cameras at indices: {found}")
        if len(found) < 2:
            print(f"ERROR: need at least 2 cameras, found {len(found)}: {found}")
            sys.exit(1)
        cam0_idx = args.cam0 if args.cam0 is not None else found[0]
        cam1_idx = args.cam1 if args.cam1 is not None else found[1]
    else:
        cam0_idx, cam1_idx = args.cam0, args.cam1

    print(f"Camera 0 → index {cam0_idx}")
    print(f"Camera 1 → index {cam1_idx}")

    cap0 = cv2.VideoCapture(cam0_idx)
    cap1 = cv2.VideoCapture(cam1_idx)

    for cap, label in [(cap0, "cam0"), (cap1, "cam1")]:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  {label}: {w}×{h} @ {fps:.1f} FPS")

    print("\nQ = quit | S = save snapshot")

    while True:
        # grab() is non-blocking; retrieve() decodes — call both grabs first
        # for the tightest possible sync between cameras.
        cap0.grab()
        cap1.grab()
        ok0, frame0 = cap0.retrieve()
        ok1, frame1 = cap1.retrieve()

        if not ok0 or not ok1:
            print("ERROR: failed to read frames")
            break

        h_target = min(frame0.shape[0], frame1.shape[0], 480)
        def resize_to_height(f: np.ndarray, h: int) -> np.ndarray:
            scale = h / f.shape[0]
            return cv2.resize(f, (int(f.shape[1] * scale), h))

        d0 = resize_to_height(frame0, h_target)
        d1 = resize_to_height(frame1, h_target)

        cv2.putText(d0, f"Camera 0  (idx {cam0_idx})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 0), 2)
        cv2.putText(d1, f"Camera 1  (idx {cam1_idx})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 0), 2)

        cv2.imshow("Step 1 — Camera Check  |  Q=quit  S=snapshot",
                   np.hstack([d0, d1]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("snapshot_cam0.jpg", frame0)
            cv2.imwrite("snapshot_cam1.jpg", frame1)
            print("Saved snapshot_cam0.jpg and snapshot_cam1.jpg")

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
