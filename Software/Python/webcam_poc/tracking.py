"""
Stereo target tracking: candidate matching, software camera sync and a 3D Kalman filter.

Two USB webcams run on independent clocks, so their frames are taken at slightly
different instants. For a moving target this puts the two detections at different
points along its path and the 3D position comes out wrong. StereoTracker:

  1. Considers the few largest moving blobs in each camera. While a track exists it
     picks the geometrically consistent pair closest to where the target is predicted
     to be; a new track only starts from a clearly consistent pair.
  2. Estimates the time offset between the cameras from the target's own motion
     (the offset that best satisfies the epipolar constraint) and, when that clearly
     helps, shifts camera 1's detection along its image velocity to camera 0's instant.
  3. Filters the 3D positions with a constant-velocity Kalman filter: smooths jitter,
     rejects impossible jumps, positions and speeds, and predicts through short dropouts.
"""
from __future__ import annotations
from collections import deque
import cv2
import numpy as np

from triangulate_utils import triangulate_point, reprojection_error

LAG_GRID_S = np.arange(-0.15, 0.1501, 0.005)
LAG_MIN_GAIN = 0.7          # apply a sync offset only if it cuts epipolar error by 30%+
LAG_UPDATE_EVERY_S = 1.0
VEL_MAX_GAP_S = 0.3
COAST_S = 0.3
GATE_D2 = 25.0              # Mahalanobis² gate (~5 sigma)
ACCEL_MM_S2 = 6000.0        # process noise: plausible hand-moved target acceleration


def _meas_sigma(z: np.ndarray) -> float:
    return 5.0 + 0.01 * float(np.linalg.norm(z))


class Kalman3D:
    """Constant-velocity Kalman filter on (x, y, z, vx, vy, vz) in mm, mm/s."""

    def __init__(self, max_speed_mm_s: float = 3000.0) -> None:
        self.x: np.ndarray | None = None
        self.P = np.eye(6)
        self.t = 0.0
        self.last_meas_t = -1e9
        self.rejects = 0
        self.max_speed = max_speed_mm_s

    @property
    def active(self) -> bool:
        return self.x is not None

    def _propagate(self, x, P, dt):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = ACCEL_MM_S2 ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * dt ** 4 / 4 * q
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * dt ** 3 / 2 * q
        Q[3:, 3:] = np.eye(3) * dt ** 2 * q
        return F @ x, F @ P @ F.T + Q

    def _clamp_speed(self) -> None:
        v = np.linalg.norm(self.x[3:])
        if v > self.max_speed:
            self.x[3:] *= self.max_speed / v

    def predicted(self, t: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Predicted position and its covariance at time t (does not change the filter)."""
        if self.x is None:
            return None
        x, P = self._propagate(self.x, self.P, max(t - self.t, 1e-3))
        return x[:3], P[:3, :3]

    def _init(self, t: float, z: np.ndarray) -> None:
        self.x = np.concatenate([z, np.zeros(3)])
        self.P = np.diag([_meas_sigma(z) ** 2] * 3 + [1000.0 ** 2] * 3)
        self.t = self.last_meas_t = t
        self.rejects = 0

    def step(self, t: float, z: np.ndarray | None, can_start: bool) -> str:
        """Advance to t with an optional measurement. Returns measured/coasting/rejected/lost."""
        if self.x is not None and t - self.last_meas_t > COAST_S:
            self.x = None
        if self.x is None:
            if z is not None and can_start:
                self._init(t, z)
                return "measured"
            return "lost"
        self.x, self.P = self._propagate(self.x, self.P, max(t - self.t, 1e-3))
        self.t = t
        self._clamp_speed()
        if z is None:
            return "coasting"
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        y = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(3) * _meas_sigma(z) ** 2
        if y @ np.linalg.solve(S, y) > GATE_D2:
            self.rejects += 1
            if self.rejects >= 3 and can_start:   # it really moved: restart there
                self._init(t, z)
                return "measured"
            return "rejected"
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self._clamp_speed()
        self.last_meas_t = t
        self.rejects = 0
        return "measured"

    @property
    def position(self) -> np.ndarray | None:
        return None if self.x is None else self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray | None:
        return None if self.x is None else self.x[3:].copy()


def estimate_lag(samples, stereo) -> tuple[float, float] | None:
    """
    Time offset (s) that best aligns camera 1 with camera 0, and how much it helps
    (median epipolar error at that offset ÷ at zero offset). Each camera-1 detection is
    shifted along its image velocity by tau. Positive tau = camera 1's frame is older.
    """
    moving = [s for s in samples if np.linalg.norm(s[2]) > 40.0]
    if len(moving) < 20:
        return None
    p0 = np.array([s[0] for s in moving], np.float64)
    p1 = np.array([s[1] for s in moving], np.float64)
    v1 = np.array([s[2] for s in moving], np.float64)
    K0, d0, K1, d1, F = stereo["K0"], stereo["dist0"], stereo["K1"], stereo["dist1"], stereo["F"]
    u0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K0, d0, P=K0).reshape(-1, 2)
    lines = np.column_stack([u0, np.ones(len(u0))]) @ F.T     # epipolar lines in camera 1
    norm = np.hypot(lines[:, 0], lines[:, 1])

    def cost(tau):
        u1 = cv2.undistortPoints((p1 + v1 * tau).reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2)
        return np.median(np.abs(np.sum(lines * np.column_stack([u1, np.ones(len(u1))]), axis=1)) / norm)

    costs = [cost(tau) for tau in LAG_GRID_S]
    i = int(np.argmin(costs))
    return float(LAG_GRID_S[i]), float(costs[i] / max(cost(0.0), 1e-9))


class StereoTracker:
    def __init__(self, stereo, max_reproj: float = 25.0, min_range_mm: float = 200.0,
                 max_range_mm: float = 4000.0, max_speed_mps: float = 3.0) -> None:
        self.s = stereo
        self.args = (stereo["K0"], stereo["dist0"], stereo["K1"], stereo["dist1"],
                     stereo["P0"], stereo["P1"])
        self.max_reproj = max_reproj
        self.range_mm = (min_range_mm, max_range_mm)
        self.kf = Kalman3D(max_speed_mps * 1000.0)
        self.lag = 0.0
        self.samples: deque = deque(maxlen=150)
        self.prev1: tuple[float, np.ndarray] | None = None
        self.next_lag_update = 0.0
        self.glitches = 0

    def _velocity1(self, t: float, c1: np.ndarray, width: float) -> np.ndarray:
        """Target's image velocity in camera 1 (px/s), from its last matched detection."""
        if self.prev1 is None or t - self.prev1[0] > VEL_MAX_GAP_S:
            return np.zeros(2)
        dp = c1 - self.prev1[1]
        if np.linalg.norm(dp) > 0.15 * width:
            return np.zeros(2)
        return dp / max(t - self.prev1[0], 1e-3)

    def _pairs(self, t, cands0, cands1, width1):
        """Every candidate pair that is geometrically consistent and physically plausible."""
        out, best_any = [], None
        for c0 in cands0:
            for c1 in cands1:
                c1 = np.asarray(c1, np.float64)
                v1 = self._velocity1(t, c1, width1)
                c1s = c1 + v1 * self.lag
                X = triangulate_point(tuple(c0), tuple(c1s), *self.args)
                if X[2] <= 0:
                    continue
                e = reprojection_error(X, tuple(c0), tuple(c1s), self.s)
                if best_any is None or e < best_any:
                    best_any = e
                rng = float(np.linalg.norm(X))
                if e <= self.max_reproj and self.range_mm[0] <= rng <= self.range_mm[1]:
                    out.append((e, X, np.asarray(c0, np.float64), c1, v1))
        return out, best_any

    def update(self, t: float, cands0: list, cands1: list, width1: float) -> dict:
        pairs, best_any = self._pairs(t, cands0, cands1, width1)
        chosen = None
        pred = self.kf.predicted(t) if self.kf.active else None
        if pairs and pred is not None:
            xp, Pp = pred

            def d2(p):
                S = Pp + np.eye(3) * _meas_sigma(p[1]) ** 2
                y = p[1] - xp
                return y @ np.linalg.solve(S, y)
            chosen = min(pairs, key=d2)
        elif pairs:
            chosen = min(pairs, key=lambda p: p[0])

        meas = None if chosen is None else chosen[1]
        strict = chosen is not None and chosen[0] <= 0.5 * self.max_reproj
        state = self.kf.step(t, meas, can_start=strict)
        if state == "rejected":
            self.glitches += 1
        if state == "measured":
            self.samples.append((chosen[2], chosen[3], chosen[4]))
            self.prev1 = (t, chosen[3])

        if t >= self.next_lag_update:
            self.next_lag_update = t + LAG_UPDATE_EVERY_S
            est = estimate_lag(list(self.samples), self.s)
            if est is not None:
                lag, gain = est
                target = lag if gain < LAG_MIN_GAIN else 0.0
                self.lag = 0.7 * self.lag + 0.3 * target

        pos, vel = self.kf.position, self.kf.velocity
        speed = closing = 0.0
        if pos is not None:
            speed = float(np.linalg.norm(vel)) / 1000.0
            closing = float(-(pos @ vel) / max(float(np.linalg.norm(pos)), 1e-6)) / 1000.0
        ok = state == "measured"
        return {"pos": pos, "raw": meas if ok else None,
                "reproj": chosen[0] if chosen is not None else best_any,
                "state": state,
                "c0": tuple(chosen[2]) if ok else None, "c1": tuple(chosen[3]) if ok else None,
                "speed": speed, "closing": closing, "lag_ms": self.lag * 1000.0,
                "no_match": bool(cands0 and cands1 and not ok and state != "coasting")}
