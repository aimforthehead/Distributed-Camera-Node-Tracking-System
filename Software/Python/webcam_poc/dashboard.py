"""
Operator console for the stereo tracker, drawn entirely with OpenCV.

Layout (1440×810, resizable window):
    ┌─────────────┬─────────────┐
    │ camera 0    │ camera 1    │
    ├─────────────┼─────────────┤
    │ top-down    │ telemetry + │
    │ radar view  │ range graph │
    └─────────────┴─────────────┘
"""
from __future__ import annotations
import math
from collections import deque
import cv2
import numpy as np

# BGR palette — dark console with a green bias, phosphor-green primary
BG = (14, 18, 13)
PANEL = (20, 26, 19)
GRID = (58, 88, 56)
GRID_FAINT = (34, 48, 33)
PHOSPHOR = (125, 225, 135)
TEXT = (212, 226, 208)
MUTED = (118, 138, 116)
AMBER = (40, 175, 245)
RED = (70, 70, 240)
CAM1_TINT = (215, 190, 110)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

W, H = 1440, 810
TW, TH = W // 2, H // 2
RANGE_SPAN_S = 15.0
SCALES_M = [0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
CLOSING_THRESHOLD = 0.10  # m/s


def _text(img, s, org, scale=0.5, color=TEXT, thick=1, font=FONT):
    cv2.putText(img, s, org, font, scale, color, thick, AA)


def _text_right(img, s, right_x, y, scale=0.5, color=TEXT, thick=1, font=FONT):
    (tw, _), _ = cv2.getTextSize(s, font, scale, thick)
    _text(img, s, (right_x - tw, y), scale, color, thick, font)


def _darken(img, y0, y1, amount=0.65):
    band = img[y0:y1].astype(np.float32)
    img[y0:y1] = (band * (1 - amount) + np.array(BG, np.float32) * amount).astype(np.uint8)


def _brackets(img, cx, cy, r, color):
    k = r // 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * r, cy + sy * r
            cv2.line(img, (x, y), (x - sx * k, y), color, 2, AA)
            cv2.line(img, (x, y), (x, y - sy * k), color, 2, AA)
    cv2.circle(img, (cx, cy), 3, color, -1, AA)


class Dashboard:
    WINDOW = "Tracking Console  |  Q quit  M mask  R reset  F fullscreen  S screenshot"

    def __init__(self, stereo) -> None:
        K0, K1, R, T = stereo["K0"], stereo["K1"], stereo["R"], stereo["T"]
        c1 = (-(R.T @ T)).flatten() / 1000.0
        self.cam1_xz = (float(c1[0]), float(c1[2]))
        w0 = float(stereo["img_size"][0])
        self.fov0 = (-math.atan(K0[0, 2] / K0[0, 0]), math.atan((w0 - K0[0, 2]) / K0[0, 0]))
        axis1 = R.T @ np.array([0.0, 0.0, 1.0])
        phi = math.atan2(axis1[0], axis1[2])
        half1 = math.atan(K1[0, 2] / K1[0, 0])  # assumes principal point near image centre
        self.fov1 = (phi - half1, phi + half1)
        self.baseline_mm = float(stereo["baseline_mm"][0])

        self.trail: deque[tuple[float, float]] = deque(maxlen=150)
        self.range_hist: deque[tuple[float, float | None]] = deque()
        self.fps = 0.0
        self.last_t: float | None = None
        self.fullscreen = False
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, W, H)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        mode = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.WINDOW, cv2.WND_PROP_FULLSCREEN, mode)

    # ------------------------------------------------------------------ render
    def render(self, t, frame0, frame1, mask0, mask1, c0, c1, pos_mm,
               speed, closing, detector, show_mask, reproj=None,
               coasting=False, lag_ms=None) -> np.ndarray:
        if self.last_t is not None and t > self.last_t:
            inst = 1.0 / (t - self.last_t)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self.last_t = t

        pos = None if pos_mm is None else np.asarray(pos_mm, float) / 1000.0
        if pos is not None:
            self.trail.append((pos[0], pos[2]))
        self.range_hist.append((t, None if pos is None else float(np.linalg.norm(pos))))
        while self.range_hist and self.range_hist[0][0] < t - RANGE_SPAN_S:
            self.range_hist.popleft()

        if pos is not None and coasting:
            status, status_color = "PREDICTING", AMBER
        elif pos is None and reproj is not None:
            status, status_color = "NO MATCH", AMBER
        elif pos is None:
            status, status_color = "SEARCHING", MUTED
        elif closing > CLOSING_THRESHOLD:
            status, status_color = "APPROACHING", RED
        elif closing < -CLOSING_THRESHOLD:
            status, status_color = "RECEDING", AMBER
        else:
            status, status_color = "TRACKING", PHOSPHOR
        target_color = RED if status == "APPROACHING" else PHOSPHOR

        canvas = np.full((H, W, 3), BG, np.uint8)
        v0 = mask0 if show_mask and mask0 is not None else frame0
        v1 = mask1 if show_mask and mask1 is not None else frame1
        canvas[:TH, :TW] = self._camera_tile(v0, c0, "CAM 0", target_color)
        canvas[:TH, TW:] = self._camera_tile(v1, c1, "CAM 1", target_color)
        canvas[TH:, :TW] = self._radar(pos, target_color, t)
        canvas[TH:, TW:] = self._telemetry(pos, speed, closing, status, status_color,
                                          c0 is not None, c1 is not None, detector, t, reproj,
                                          lag_ms)

        cv2.line(canvas, (TW, 0), (TW, H), GRID_FAINT, 2)
        cv2.line(canvas, (0, TH), (W, TH), GRID_FAINT, 2)
        if status == "APPROACHING" and int(t * 4) % 2 == 0:
            cv2.rectangle(canvas, (2, 2), (W - 3, H - 3), RED, 4)
        return canvas

    # ------------------------------------------------------------- components
    def _camera_tile(self, frame, center, label, target_color) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        tile = np.full((TH, TW, 3), BG, np.uint8)
        h, w = frame.shape[:2]
        s = min(TW / w, TH / h)
        nw, nh = int(w * s), int(h * s)
        ox, oy = (TW - nw) // 2, (TH - nh) // 2
        tile[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh))
        if center is not None:
            cx, cy = int(ox + center[0] * s), int(oy + center[1] * s)
            _brackets(tile, cx, cy, 24, target_color)
            _text(tile, f"px ({center[0]:.0f}, {center[1]:.0f})", (cx + 30, cy - 28),
                  0.45, target_color)
        _darken(tile, 0, 32)
        _text(tile, label, (12, 22), 0.6, TEXT, 1, FONT_B)
        if center is not None:
            cv2.circle(tile, (TW - 86, 16), 5, PHOSPHOR, -1, AA)
            _text(tile, "LOCK", (TW - 74, 22), 0.55, PHOSPHOR, 1, FONT_B)
        else:
            cv2.circle(tile, (TW - 100, 16), 5, MUTED, 1, AA)
            _text(tile, "SEARCH", (TW - 88, 22), 0.55, MUTED, 1, FONT_B)
        return tile

    def _radar(self, pos, target_color, t) -> np.ndarray:
        img = np.full((TH, TW, 3), PANEL, np.uint8)
        ox, oy = TW // 2, TH - 44
        radius = TH - 90

        extent = max([math.hypot(*self.cam1_xz)] +
                     [math.hypot(x, z) for x, z in self.trail] + [0.4]) * 1.15
        scale = next((s for s in SCALES_M if s >= extent), SCALES_M[-1])
        ppm = radius / scale

        def to_px(x, z):
            return int(ox + x * ppm), int(oy - z * ppm)

        def bearing_pt(origin, theta, r):
            return int(origin[0] + r * math.sin(theta)), int(origin[1] - r * math.cos(theta))

        # Camera coverage wedges; tracking works where they overlap
        overlay = img.copy()
        for (cx, cz), (a, b), col in [((0.0, 0.0), self.fov0, PHOSPHOR),
                                     (self.cam1_xz, self.fov1, CAM1_TINT)]:
            o = to_px(cx, cz)
            pts = [o] + [bearing_pt(o, a + (b - a) * i / 24, radius * 1.3) for i in range(25)]
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], col, AA)
        img = cv2.addWeighted(overlay, 0.07, img, 0.93, 0)

        for i in range(1, 5):
            r = int(radius * i / 4)
            cv2.ellipse(img, (ox, oy), (r, r), 0, 180, 360, GRID if i == 4 else GRID_FAINT, 1, AA)
            _text(img, f"{scale * i / 4:g} m", (ox + r + 4, oy - 4), 0.4, MUTED)
        for deg in range(-90, 91, 30):
            cv2.line(img, (ox, oy), bearing_pt((ox, oy), math.radians(deg), radius), GRID_FAINT, 1, AA)
            if deg not in (-90, 90):
                p = bearing_pt((ox, oy), math.radians(deg), radius + 14)
                _text(img, f"{deg:+d}", (p[0] - 12, p[1]), 0.38, MUTED)

        # Sweep line oscillating across the forward arc
        sweep = math.radians(80 * math.sin(t * 1.2))
        glow = img.copy()
        for k in range(3):
            cv2.line(glow, (ox, oy), bearing_pt((ox, oy), sweep - k * 0.03, radius), PHOSPHOR, 1, AA)
        img = cv2.addWeighted(glow, 0.3, img, 0.7, 0)

        # Cameras
        for (cx, cz), col, name in [((0.0, 0.0), PHOSPHOR, "C0"), (self.cam1_xz, CAM1_TINT, "C1")]:
            px, py = to_px(cx, cz)
            tri = np.array([[px, py - 9], [px - 8, py + 6], [px + 8, py + 6]], np.int32)
            cv2.fillPoly(img, [tri], col, AA)
            _text(img, name, (px - 8, py + 22), 0.4, col)

        # Trail and target
        n = len(self.trail)
        for i, (x, z) in enumerate(self.trail):
            a = (i + 1) / n
            col = tuple(int(c * (0.2 + 0.6 * a)) for c in PHOSPHOR)
            cv2.circle(img, to_px(x, z), 2, col, -1, AA)
        if pos is not None:
            p = to_px(pos[0], pos[2])
            cv2.line(img, (ox, oy), p, target_color, 1, AA)
            cv2.circle(img, p, 7, target_color, -1, AA)
            cv2.circle(img, p, 13 + int(4 * (1 + math.sin(t * 8))), target_color, 1, AA)
            _text(img, f"TGT {np.linalg.norm(pos):.2f} m", (p[0] + 16, p[1] - 10), 0.5, target_color, 1, FONT_B)

        _text(img, "TOP-DOWN VIEW", (12, 22), 0.6, TEXT, 1, FONT_B)
        _text(img, "X = left/right   Z = forward from CAM 0", (12, 42), 0.42, MUTED)
        _text_right(img, f"scale {scale:g} m", TW - 12, 22, 0.45, MUTED)
        return img

    def _telemetry(self, pos, speed, closing, status, status_color,
                   lock0, lock1, detector, t, reproj=None, lag_ms=None) -> np.ndarray:
        img = np.full((TH, TW, 3), PANEL, np.uint8)
        _text(img, "TRACK 01", (20, 30), 0.7, TEXT, 1, FONT_B)

        (sw, sh), _ = cv2.getTextSize(status, FONT_B, 0.65, 1)
        x1 = TW - 20
        x0 = x1 - sw - 28
        if status == "APPROACHING":
            cv2.rectangle(img, (x0, 10), (x1, 40), status_color, -1)
            _text(img, status, (x0 + 14, 33), 0.65, (255, 255, 255), 1, FONT_B)
        else:
            cv2.rectangle(img, (x0, 10), (x1, 40), status_color, 1)
            _text(img, status, (x0 + 14, 33), 0.65, status_color, 1, FONT_B)

        sys_line = (f"CAM0 {'LOCK' if lock0 else '----'}   CAM1 {'LOCK' if lock1 else '----'}   "
                    f"{self.fps:4.1f} FPS   {detector.upper()}   BASELINE {self.baseline_mm:.0f} mm   "
                    f"MATCH ERR {'--' if reproj is None else f'{reproj:.1f} px'}"
                    f"{'' if lag_ms is None else f'   SYNC {lag_ms:+.0f} ms'}")
        _text(img, sys_line, (20, 60), 0.42, MUTED)
        cv2.line(img, (20, 72), (TW - 20, 72), GRID_FAINT, 1)

        if pos is not None:
            x, y, z = pos
            height = -y
            rng = float(np.linalg.norm(pos))
            bearing = math.degrees(math.atan2(x, z))
            elevation = math.degrees(math.atan2(height, math.hypot(x, z)))
            rows = [
                ("X  LEFT / RIGHT", f"{x:+.2f} m", "RANGE", f"{rng:.2f} m"),
                ("Y  HEIGHT", f"{height:+.2f} m", "SPEED", f"{speed:.2f} m/s"),
                ("Z  FORWARD", f"{z:.2f} m", "CLOSING", f"{closing:+.2f} m/s"),
                ("BEARING", f"{bearing:+.1f} deg", "ELEVATION", f"{elevation:+.1f} deg"),
            ]
        else:
            rows = [("X  LEFT / RIGHT", "--", "RANGE", "--"),
                    ("Y  HEIGHT", "--", "SPEED", "--"),
                    ("Z  FORWARD", "--", "CLOSING", "--"),
                    ("BEARING", "--", "ELEVATION", "--")]

        for i, (l1, v1, l2, v2) in enumerate(rows):
            y0 = 96 + i * 50
            for col_x, label, value in [(20, l1, v1), (TW // 2 + 10, l2, v2)]:
                _text(img, label, (col_x, y0), 0.4, MUTED)
                vcol = status_color if label == "CLOSING" and pos is not None else TEXT
                _text(img, value, (col_x, y0 + 26), 0.8, vcol, 1, FONT_B)

        self._range_graph(img, 20, 312, TW - 40, 78, t)
        return img

    def _range_graph(self, img, x0, y0, w, h, t) -> None:
        _text(img, f"RANGE, LAST {RANGE_SPAN_S:.0f} s", (x0, y0 - 8), 0.4, MUTED)
        for i in range(3):
            yy = y0 + int(h * i / 2)
            cv2.line(img, (x0, yy), (x0 + w, yy), GRID_FAINT, 1)
        vals = [r for _, r in self.range_hist if r is not None]
        if not vals:
            _text(img, "no track yet", (x0 + 8, y0 + h // 2 + 5), 0.45, MUTED)
            return
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.15, 0.05)
        lo, hi = lo - pad, hi + pad

        segs, cur = [], []
        for tt, r in self.range_hist:
            if r is None:
                if cur:
                    segs.append(cur)
                cur = []
                continue
            px = x0 + int((tt - (t - RANGE_SPAN_S)) / RANGE_SPAN_S * w)
            py = y0 + h - int((r - lo) / (hi - lo) * h)
            cur.append((px, py))
        if cur:
            segs.append(cur)
        for seg in segs:
            if len(seg) > 1:
                cv2.polylines(img, [np.array(seg, np.int32)], False, PHOSPHOR, 2, AA)
        if self.range_hist[-1][1] is not None and segs:
            cv2.circle(img, segs[-1][-1], 4, PHOSPHOR, -1, AA)
        _text(img, f"{hi:.2f} m", (x0 + 4, y0 + 14), 0.38, MUTED)
        _text(img, f"{lo:.2f} m", (x0 + 4, y0 + h - 4), 0.38, MUTED)
