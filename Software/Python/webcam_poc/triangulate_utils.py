"""
Shared utilities for the webcam triangulation POC.
Imported by 05_triangulate.py and 06_accuracy_check.py.
"""
from __future__ import annotations
import json
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
