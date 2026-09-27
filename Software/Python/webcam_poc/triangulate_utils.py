"""
Shared utilities for the webcam triangulation POC.
Imported by 05_triangulate.py and 06_accuracy_check.py.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np


def load_calibration(calib_dir: Path, need_hsv: bool = True) -> tuple[dict, dict | None]:
    """Load stereo calibration and (optionally) HSV params."""
    stereo_file = calib_dir / "stereo.npz"
    hsv_file = calib_dir / "hsv_params.json"
    if not stereo_file.exists():
        raise FileNotFoundError(
            f"{stereo_file} not found — run 02_calibrate_intrinsics.py then 03_calibrate_extrinsics.py first"
        )
    stereo = np.load(stereo_file)
    if not need_hsv:
        return stereo, None
    if not hsv_file.exists():
        raise FileNotFoundError(
            f"{hsv_file} not found — run 04_detect_object.py first"
        )
    with open(hsv_file) as f:
        hsv = json.load(f)
    return stereo, hsv


class MotionDetector:
    """
    Detects the largest moving object against a static background (sky, wall, ceiling).

    Uses MOG2 background subtraction: each pixel keeps a statistical model of its
    usual value; pixels that suddenly differ are flagged as foreground. Needs a
    fixed camera. One instance per camera, because each keeps its own background model.
    """

    def __init__(self, min_area: int = 80, history: int = 300, var_threshold: float = 32):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, tuple | None]:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self.bg.apply(blurred)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # Merge fragments of one object (propellers, arms) into a single blob
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        if not contours:
            return mask, None
        best = max(contours, key=cv2.contourArea)
        m = cv2.moments(best)
        return mask, (m["m10"] / m["m00"], m["m01"] / m["m00"])


def detect_ball(frame: np.ndarray, hsv_params: dict) -> tuple[np.ndarray | None, tuple | None]:
    """
    Detect a colored ball using HSV thresholding.

    Returns (mask, center_xy) where center_xy is (float, float) in pixel space,
    or (mask, None) if no ball found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([hsv_params["h_low"], hsv_params["s_low"], hsv_params["v_low"]])
    hi = np.array([hsv_params["h_high"], hsv_params["s_high"], hsv_params["v_high"]])
    mask = cv2.inRange(hsv, lo, hi)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 150:
        return mask, None

    (cx, cy), _ = cv2.minEnclosingCircle(best)
    return mask, (float(cx), float(cy))


def triangulate_point(
    pt0: tuple[float, float],
    pt1: tuple[float, float],
    K0: np.ndarray, dist0: np.ndarray,
    K1: np.ndarray, dist1: np.ndarray,
    P0: np.ndarray, P1: np.ndarray,
) -> np.ndarray:
    """
    Triangulate a 3D world point from two 2D pixel observations.

    Steps:
      1. Undistort both pixel points: remove lens distortion and remap to
         corrected pixel coords (undistortPoints with P=K applies K⁻¹ then K,
         net effect is just distortion removal in pixel space).
      2. DLT triangulation via cv2.triangulatePoints — solves the 4×4 system
         formed by cross-multiplying p = P*X for both cameras.
      3. Dehomogenize the result: divide XYZ by W.

    Args:
        pt0, pt1: raw pixel coords (u, v) in each camera
        K0/K1: 3×3 intrinsic matrices
        dist0/dist1: distortion coefficient arrays
        P0/P1: 3×4 projection matrices (K·[R|t]) — P0 = K0·[I|0]

    Returns:
        (X, Y, Z) in mm in camera-0 world frame (cam0 = origin, +Z forward)
    """
    p0 = np.array([[[pt0[0], pt0[1]]]], dtype=np.float32)
    p1 = np.array([[[pt1[0], pt1[1]]]], dtype=np.float32)

    # Undistort: corrected pixel coords (distortion removed, still in pixel space)
    p0_ud = cv2.undistortPoints(p0, K0, dist0, P=K0)  # shape (1,1,2)
    p1_ud = cv2.undistortPoints(p1, K1, dist1, P=K1)

    # triangulatePoints expects (2, N) arrays
    pt0_arr = p0_ud.reshape(2, 1).astype(np.float64)
    pt1_arr = p1_ud.reshape(2, 1).astype(np.float64)

    X_hom = cv2.triangulatePoints(
        P0.astype(np.float64),
        P1.astype(np.float64),
        pt0_arr,
        pt1_arr,
    )  # 4×1 homogeneous
    X = X_hom[:3] / X_hom[3]
    return X.flatten()


def reprojection_error(X: np.ndarray, pt0: tuple, pt1: tuple, stereo) -> float:
    """
    Project the 3D point back into both cameras and return the worse pixel miss.

    If both detections really are the same object, the triangulated point lands on
    top of both of them (a few px, limited by calibration). A large value means the
    cameras locked onto different things (e.g. the drone in one view, a hand in the other).
    """
    obj = np.asarray(X, np.float64).reshape(1, 1, 3)
    zero = np.zeros(3)
    p0, _ = cv2.projectPoints(obj, zero, zero, stereo["K0"], stereo["dist0"])
    rvec1, _ = cv2.Rodrigues(stereo["R"])
    p1, _ = cv2.projectPoints(obj, rvec1, stereo["T"], stereo["K1"], stereo["dist1"])
    e0 = np.linalg.norm(p0.ravel() - np.asarray(pt0, float))
    e1 = np.linalg.norm(p1.ravel() - np.asarray(pt1, float))
    return float(max(e0, e1))


def print_rig_geometry(R: np.ndarray, T: np.ndarray) -> None:
    """Print the measured camera layout: convergence, height difference, where the axes cross."""
    c1 = (-R.T @ T).ravel()
    axis1 = R.T @ np.array([0.0, 0.0, 1.0])
    convergence = np.degrees(np.arccos(np.clip(axis1[2], -1.0, 1.0)))
    yaw1 = np.degrees(np.arctan2(axis1[0], axis1[2]))
    print(f"  Convergence      : {convergence:.1f}° between the two optical axes "
          f"(cam1 yawed {yaw1:+.1f}° relative to cam0)")
    print(f"  Height difference: {-c1[1]:+.0f} mm (cam1 relative to cam0, + = higher)")
    d0, d1, w = np.array([0.0, 0.0, 1.0]), axis1, -c1
    a, b, c, d, e = d0 @ d0, d0 @ d1, d1 @ d1, d0 @ w, d1 @ w
    denom = a * c - b * b
    if denom > 1e-9:
        s = (b * e - c * d) / denom
        u = (a * e - b * d) / denom
        aim = (s * d0 + c1 + u * d1) / 2
        if aim[2] > 0:
            print(f"  Axes cross at    : {aim[2] / 10:.0f} cm in front of cam0")


def beep() -> None:
    """Audible confirmation, so the person holding the board knows a capture happened."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)
    except OSError:
        pass


class AutoCapture:
    """
    Hands-free capture: fires when the board has been held still for `hold_s` seconds
    at a position that differs enough from every previous capture.

    Feed it the detected corners of every camera (or None if any camera lacks the board).
    Corners are normalised by image width so thresholds work at any resolution.
    """

    def __init__(self, hold_s: float = 0.8, still_tol: float = 0.004,
                 min_change: float = 0.05, cooldown_s: float = 1.0):
        self.hold_s, self.still_tol = hold_s, still_tol
        self.min_change, self.cooldown_s = min_change, cooldown_s
        self.prev = None
        self.still_since = None
        self.last_capture = -1e9
        self.shots: list[np.ndarray] = []

    def update(self, corner_sets, widths, t: float) -> tuple[bool, str, float]:
        """Returns (capture_now, state, hold_progress); state is none/move/hold/captured."""
        if corner_sets is None:
            self.prev = self.still_since = None
            return False, "none", 0.0
        c = np.vstack([cs.reshape(-1, 2) / w for cs, w in zip(corner_sets, widths)])
        still = (self.prev is not None and self.prev.shape == c.shape and
                 np.mean(np.linalg.norm(c - self.prev, axis=1)) < self.still_tol)
        if still:
            if self.still_since is None:
                self.still_since = t
        else:
            self.still_since = None
        self.prev = c
        if t - self.last_capture < self.cooldown_s:
            return False, "captured", 1.0
        if any(np.mean(np.linalg.norm(c - s, axis=1)) < self.min_change for s in self.shots):
            return False, "move", 0.0
        if self.still_since is None:
            return False, "hold", 0.0
        progress = min((t - self.still_since) / self.hold_s, 1.0)
        if progress >= 1.0:
            self.shots.append(c)
            self.last_capture = t
            self.still_since = None
            return True, "captured", 1.0
        return False, "hold", progress


def draw_auto_status(img: np.ndarray, state: str, progress: float, n: int, target: int) -> None:
    """Large status band at the bottom of a view, readable from a couple of metres away."""
    h, w = img.shape[:2]
    sc = max(0.8, w / 1280)
    text, color = {
        "none": ("SHOW THE BOARD", (0, 0, 230)),
        "move": ("MOVE OR TILT TO A NEW POSITION", (0, 200, 255)),
        "hold": ("HOLD STILL...", (0, 230, 230)),
        "captured": (f"CAPTURED {n}/{target}", (0, 230, 0)),
    }[state]
    band = int(100 * sc)
    img[h - band:] = (img[h - band:] * 0.3).astype(np.uint8)
    cv2.putText(img, text, (int(20 * sc), h - int(50 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                1.3 * sc, color, max(2, int(3 * sc)), cv2.LINE_AA)
    label = f"{n}/{target}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3 * sc, 3)
    cv2.putText(img, label, (w - tw - int(20 * sc), h - int(50 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                1.3 * sc, (230, 230, 230), max(2, int(3 * sc)), cv2.LINE_AA)
    bar_w = int((w - 40 * sc) * (progress if state == "hold" else (1.0 if state == "captured" else 0)))
    y = h - int(25 * sc)
    cv2.rectangle(img, (int(20 * sc), y), (int(20 * sc) + bar_w, y + int(10 * sc)), color, -1)
    if state == "captured":
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 230, 0), max(6, int(12 * sc)))


def check_frame_size(cap, expected, label: str) -> None:
    """Stop early if a camera's resolution differs from the one it was calibrated at."""
    ok, frame = cap.read()
    if not ok:
        print(f"ERROR: {label} gives no image — is it unplugged?")
        sys.exit(1)
    h, w = frame.shape[:2]
    ew, eh = (int(v) for v in expected)
    if (w, h) != (ew, eh):
        print(f"ERROR: {label} gives {w}x{h} but its calibration is for {ew}x{eh}.")
        print("  The camera numbers probably changed (replug). Run 01_check_cameras.py,")
        print("  find which index is which camera, and pass the right --cam0/--cam1.")
        sys.exit(1)
