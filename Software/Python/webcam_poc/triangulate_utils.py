"""
Shared utilities for the webcam triangulation POC.
Imported by 05_triangulate.py and 06_accuracy_check.py.
"""
import json
from pathlib import Path
import cv2
import numpy as np


def load_calibration(calib_dir: Path) -> tuple[dict, dict]:
    """Load stereo calibration and HSV params. Raises FileNotFoundError with clear message."""
    stereo_file = calib_dir / "stereo.npz"
    hsv_file = calib_dir / "hsv_params.json"
    if not stereo_file.exists():
        raise FileNotFoundError(
            f"{stereo_file} not found — run 02_calibrate_intrinsics.py then 03_calibrate_extrinsics.py first"
        )
    if not hsv_file.exists():
        raise FileNotFoundError(
            f"{hsv_file} not found — run 04_detect_object.py first"
        )
    stereo = np.load(stereo_file)
    with open(hsv_file) as f:
        hsv = json.load(f)
    return stereo, hsv


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
         formed by cross-multiplying p = P*X for both cameras (see MATH.md).
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
