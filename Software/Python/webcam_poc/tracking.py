"""
Stereo target tracking: candidate matching, software camera sync and a 3D Kalman filter.

Two USB webcams run on independent clocks, so their frames are taken at slightly
different instants. For a moving target this puts the two detections at different
points along its path and the 3D position comes out wrong. StereoTracker:

  1. Considers the few largest moving blobs in each camera and picks the pair that
     is geometrically consistent (so a hand in one view doesn't pair with the drone
     in the other).
  2. Estimates the time offset between the cameras from the target's own motion
     (the offset that best satisfies the epipolar constraint) and shifts camera 1's
     detection along its image velocity to line it up with camera 0's instant.
  3. Filters the 3D positions with a constant-velocity Kalman filter: smooths jitter,
     rejects physically impossible jumps, and predicts through short dropouts.
"""
from __future__ import annotations
from collections import deque
import cv2
import numpy as np

from triangulate_utils import triangulate_point, reprojection_error

LAG_GRID_S = np.arange(-0.25, 0.2501, 0.005)
LAG_UPDATE_EVERY_S = 1.0
VEL_MAX_GAP_S = 0.3
COAST_S = 0.5
GATE_D2 = 25.0              # Mahalanobis² gate (~5 sigma)
ACCEL_MM_S2 = 6000.0        # process noise: plausible hand-moved target acceleration


class Kalman3D:
    """Constant-velocity Kalman filter on (x, y, z, vx, vy, vz) in mm, mm/s."""

    def __init__(self) -> None:
        self.x: np.ndarray | None = None
        self.P = np.eye(6)
        self.t = 0.0
        self.last_meas_t = -1e9
        self.rejects = 0

    def _predict(self, t: float) -> None:
        dt = max(t - self.t, 1e-3)
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = ACCEL_MM_S2 ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * dt ** 4 / 4 * q
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * dt ** 3 / 2 * q
        Q[3:, 3:] = np.eye(3) * dt ** 2 * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def _init(self, t: float, z: np.ndarray, sigma: float) -> None:
        self.x = np.concatenate([z, np.zeros(3)])
        self.P = np.diag([sigma ** 2] * 3 + [1000.0 ** 2] * 3)
        self.t = self.last_meas_t = t
        self.rejects = 0

    def step(self, t: float, z: np.ndarray | None) -> str:
        """Advance to time t with an optional measurement. Returns measured/coasting/rejected/lost."""
        if z is not None:
            sigma = 5.0 + 0.01 * float(np.linalg.norm(z))
        if self.x is None or t - self.last_meas_t > COAST_S:
            if z is None:
                self.x = None
                return "lost"
            self._init(t, z, sigma)
            return "measured"
        self._predict(t)
        if z is None:
            return "coasting"
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        y = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(3) * sigma ** 2
        if y @ np.linalg.solve(S, y) > GATE_D2:
            self.rejects += 1
            if self.rejects >= 3:              # it really moved: restart on the new position
                self._init(t, z, sigma)
                return "measured"
            return "rejected"
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.last_meas_t = t
        self.rejects = 0
        return "measured"

    @property
    def position(self) -> np.ndarray | None:
        return None if self.x is None else self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray | None:
        return None if self.x is None else self.x[3:].copy()


def estimate_lag(samples, stereo) -> float | None:
    """
    Time offset (s) that best aligns camera 1 with camera 0: shift each camera-1 detection
    along its image velocity by tau and pick the tau with the smallest median epipolar
    distance. Positive tau = camera 1's frame was taken earlier.
    """
    moving = [s for s in samples if np.linalg.norm(s[2]) > 40.0]
    if len(moving) < 20:
        return None
    p0 = np.array([s[0] for s in moving], np.float64)
    p1 = np.array([s[1] for s in moving], np.float64)
    v1 = np.array([s[2] for s in moving], np.float64)
    K0, d0, K1, d1, F = stereo["K0"], stereo["dist0"], stereo["K1"], stereo["dist1"], stereo["F"]
    u0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K0, d0, P=K0).reshape(-1, 2)
    x0 = np.column_stack([u0, np.ones(len(u0))])
    lines = x0 @ F.T                                    # epipolar lines in camera 1
    norm = np.hypot(lines[:, 0], lines[:, 1])
    best, best_cost = None, np.inf
    for tau in LAG_GRID_S:
        u1 = cv2.undistortPoints((p1 + v1 * tau).reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2)
        d = np.abs(np.sum(lines * np.column_stack([u1, np.ones(len(u1))]), axis=1)) / norm
        cost = np.median(d)
        if cost < best_cost:
            best, best_cost = float(tau), cost
    return best


class StereoTracker:
    def __init__(self, stereo, max_reproj: float = 25.0) -> None:
        self.s = stereo
        self.args = (stereo["K0"], stereo["dist0"], stereo["K1"], stereo["dist1"],
                     stereo["P0"], stereo["P1"])
        self.max_reproj = max_reproj
        self.kf = Kalman3D()
        self.lag = 0.0
        self.samples: deque = deque(maxlen=150)
        self.prev1: tuple[float, np.ndarray] | None = None
        self.next_lag_update = 0.0
        self.glitches = 0

    def _velocity1(self, t: float, c1: np.ndarray, width: float) -> np.ndarray:
        if self.prev1 is None or t - self.prev1[0] > VEL_MAX_GAP_S:
            return np.zeros(2)
        dp = c1 - self.prev1[1]
        if np.linalg.norm(dp) > 0.15 * width:
            return np.zeros(2)
        return dp / max(t - self.prev1[0], 1e-3)

    def update(self, t: float, cands0: list, cands1: list, width1: float) -> dict:
        best = None
        for c0 in cands0:
            for c1 in cands1:
                c1 = np.asarray(c1, np.float64)
                v1 = self._velocity1(t, c1, width1)
                c1s = c1 + v1 * self.lag
                X = triangulate_point(tuple(c0), tuple(c1s), *self.args)
                if X[2] <= 0:
                    continue
                e = reprojection_error(X, tuple(c0), tuple(c1s), self.s)
                if best is None or e < best[0]:
                    best = (e, X, np.asarray(c0, np.float64), c1, v1)

        meas, reproj = None, None
        if best is not None:
            reproj = best[0]
            if best[0] <= self.max_reproj:
                meas = best[1]
                self.samples.append((best[2], best[3], best[4]))
                self.prev1 = (t, best[3])

        state = self.kf.step(t, meas)
        if state == "rejected":
            self.glitches += 1

        if t >= self.next_lag_update:
            self.next_lag_update = t + LAG_UPDATE_EVERY_S
            lag = estimate_lag(list(self.samples), self.s)
            if lag is not None:
                self.lag = lag if self.lag == 0.0 else 0.7 * self.lag + 0.3 * lag

        pos, vel = self.kf.position, self.kf.velocity
        speed = closing = 0.0
        if pos is not None:
            speed = float(np.linalg.norm(vel)) / 1000.0
            rng = float(np.linalg.norm(pos))
            closing = float(-(pos @ vel) / max(rng, 1e-6)) / 1000.0
        c0 = c1 = None
        if best is not None and meas is not None:
            c0, c1 = tuple(best[2]), tuple(best[3])
        return {"pos": pos, "raw": meas, "reproj": reproj, "state": state, "c0": c0, "c1": c1,
                "speed": speed, "closing": closing, "lag_ms": self.lag * 1000.0,
                "no_match": best is not None and meas is None}
