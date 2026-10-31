"""
Controls
  hands  : touch your PINKY tip to your THUMB tip (either hand) -> next filter
  n / SPACE: next filter        p: previous filter
  c        : switch camera (your choice is remembered for next time)
  s        : split mode (two filters in one frame)
  m        : mirror on/off      h: help on/off
  q / ESC  : quit
"""

import argparse
import json
import subprocess
import math
import os
import sys
import time
import urllib.request

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
SETTINGS_PATH = os.path.join(HERE, "portal_settings.json")

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_TIP, PINKY_TIP, MIDDLE_MCP = 0, 4, 8, 20, 9


def now():
    return time.monotonic()


# filters: 
_grid_cache = {}


def _grid(h, w):
    key = (h, w)
    if key not in _grid_cache:
        if len(_grid_cache) > 20:
            _grid_cache.clear()
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        _grid_cache[key] = (xs, ys)
    return _grid_cache[key]


def _gradient_lut(stops):
    pos = np.array([s[0] for s in stops]) * 255
    cols = np.array([s[1] for s in stops], np.float32)
    x = np.arange(256)
    return np.stack([np.interp(x, pos, cols[:, c]) for c in range(3)], 1).astype(np.uint8)


# night navy -> deep teal -> cyan -> copper -> warm white   (BGR)
ETHEREAL = _gradient_lut([(0.00, (10, 3, 1)), (0.30, (50, 24, 5)), (0.50, (105, 78, 12)),
                          (0.68, (150, 138, 45)), (0.84, (70, 128, 210)), (1.00, (205, 232, 255))])
# glow colours: teal -> cyan -> copper -> gold
GLOW = _gradient_lut([(0.0, (120, 90, 10)), (0.35, (230, 210, 60)), (0.65, (40, 120, 235)),
                      (1.0, (150, 225, 255))])


def _hsv_to_bgr(h, s, v):
    hsv = np.dstack([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _screen(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return (255 - (255 - a) * (255 - b) / 255).astype(np.uint8)


def _bloom(img, sigma=10, strength=0.8, thresh=150):
    bright = cv2.threshold(img, thresh, 255, cv2.THRESH_TOZERO)[1]
    glow = cv2.GaussianBlur(bright, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0, glow, strength, 0)


def ethereal_grade(img, keep_color=0.18):
    """Teal-and-copper night grade used as the base of every filter."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3)).apply(gray)
    graded = ETHEREAL[gray]
    return cv2.addWeighted(graded, 1 - keep_color, img, keep_color, 0)


def _iridescent(h, w, t, xs=None, ys=None, speed=60, scale=0.25):
    if xs is None:
        xs, ys = _grid(h, w)
    hue = ((xs * 0.6 + ys) * scale + t * speed) % 180
    return _hsv_to_bgr(hue, np.full((h, w), 170), np.full((h, w), 255))


def f_ethereal(img):
    return _bloom(ethereal_grade(img), sigma=14, strength=0.7, thresh=150)


def f_halo(img):
    t = now()
    h, w = img.shape[:2]
    base = ethereal_grade(img)
    centre = (w / 2, h / 2)
    R = math.hypot(w, h) / 2
    nr, na = 256, 720
    polar = cv2.warpPolar(base, (nr, na), centre, R, cv2.WARP_POLAR_LINEAR).astype(np.float32)
    streak = cv2.blur(polar, (31, 1))  # zoom blur outwards
    # fine fibres: each angle gets its own flickering brightness
    rng = np.random.default_rng(7)
    phase = rng.uniform(0, 2 * np.pi, na)
    freq = rng.uniform(0.5, 2.0, na)
    gain = (0.5 + 0.5 * np.sin(phase + t * freq)) ** 3
    radial = np.linspace(0, 1, nr, dtype=np.float32)
    rim = np.clip((radial - 0.15) * 1.5, 0, 1) ** 1.5
    col = GLOW[np.clip(radial * 230 + 25 * math.sin(t), 0, 255).astype(np.uint8)].astype(np.float32)
    fib = gain[:, None, None] * rim[None, :, None] * col[None, :, :] * 0.8
    polar_out = np.clip(streak * (0.55 + 0.6 * gain[:, None, None]) + fib, 0, 255).astype(np.uint8)
    out = cv2.warpPolar(polar_out, (w, h), centre, R,
                        cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP)
    out = cv2.addWeighted(base, 0.55, out, 0.6, 0)
    xs, ys = _grid(h, w)
    d = np.hypot(xs - centre[0], ys - centre[1]) / (0.05 * min(w, h) + 1)
    core = np.exp(-d * d)[..., None] * np.array([255, 245, 225], np.float32)
    return _bloom(cv2.add(out, core.astype(np.uint8)), 8, 0.4, 190)


def f_beam(img):
    t = now()
    h, w = img.shape[:2]
    xr = np.arange(w, dtype=np.float32)[None, :, None]
    yr = np.arange(h, dtype=np.float32)[:, None, None]
    out = ethereal_grade(img, keep_color=0.12).astype(np.float32) * 0.8
    cx = w / 2 + 3 * math.sin(t * 1.3)
    dx = xr - cx
    pulse = 0.85 + 0.15 * math.sin(t * 4)
    core = np.exp(-(dx / 2.5) ** 2) * np.array([255, 250, 240], np.float32)
    halo = np.exp(-(dx / (w * 0.06)) ** 2) * 0.7 * np.array([255, 220, 90], np.float32)
    wide = np.exp(-(dx / (w * 0.22)) ** 2) * 0.25 * np.array([200, 140, 40], np.float32)
    flicker = 0.75 + 0.25 * np.sin(yr / 9 - t * 12)
    beam = (core + wide) * pulse + halo * flicker * pulse
    # flattened glowing ring (a disc of light the beam passes through)
    ry = h * (0.65 + 0.08 * math.sin(t * 0.7))
    ring = np.exp(-(dx / (w * 0.32)) ** 2) * np.exp(-((yr - ry) / (h * 0.025)) ** 2)
    beam = beam + ring * np.array([90, 170, 255], np.float32) * 1.3
    out = np.clip(out + beam, 0, 255).astype(np.uint8)
    return _bloom(out, 10, 0.5, 170)


def f_chrome(img):
    t = now()
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy)  # 0..2pi, behaves like a surface normal
    n = g / 255.0
    # metallic banded reflections, brighter where the scene is lit
    val = (0.5 + 0.5 * np.cos(n * np.pi * 3 + t * 0.8)) ** 2 * (0.25 + 0.95 * n)
    # holographic thin-film colour driven by the surface "normal"
    hue = (ang * 180 / np.pi * 0.5 + n * 120 + t * 35) % 180
    sat = np.clip(mag * 9 + 70, 70, 220)
    holo = _hsv_to_bgr(hue, sat, np.full_like(sat, 255)).astype(np.float32)
    out = holo * val[..., None]
    out = np.maximum(out, ETHEREAL[g.astype(np.uint8)].astype(np.float32) * 0.45)
    out = np.clip(out, 0, 255).astype(np.uint8)
    spec = cv2.threshold(g.astype(np.uint8), 200, 255, cv2.THRESH_TOZERO)[1]
    out = cv2.add(out, cv2.cvtColor(cv2.GaussianBlur(spec, (0, 0), 4), cv2.COLOR_GRAY2BGR))
    return out


_STAR = None


def _star_kernel(size=15):
    global _STAR
    if _STAR is None:
        k = np.zeros((size, size), np.float32)
        c = size // 2
        for i in range(size):
            fall = math.exp(-abs(i - c) / (size / 6))
            k[c, i] = max(k[c, i], fall)
            k[i, c] = max(k[i, c], fall)
            if 0 <= i < size:
                k[i, i] = max(k[i, i], fall * 0.35)
                k[i, size - 1 - i] = max(k[i, size - 1 - i], fall * 0.35)
        _STAR = k
    return _STAR


def f_glitter(img):
    t = now()
    h, w = img.shape[:2]
    base = ethereal_grade(img, keep_color=0.25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # wet look: extra local contrast + specular glow
    wet = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6)).apply(gray)
    spec = cv2.threshold(wet, 185, 255, cv2.THRESH_TOZERO)[1]
    base = cv2.add(base, cv2.cvtColor(cv2.GaussianBlur(spec, (0, 0), 3), cv2.COLOR_GRAY2BGR))
    # glitter points, denser on bright areas, reshuffled a few times per second
    rng = np.random.default_rng(int(t * 8))
    n = max(20, (h * w) // 600)
    px = rng.integers(0, w, n)
    py = rng.integers(0, h, n)
    keep = rng.random(n) < (gray[py, px] / 255.0) ** 1.5 + 0.08
    px, py = px[keep], py[keep]
    sparkle = np.zeros((h, w, 3), np.uint8)
    sizes = rng.uniform(2, 9, len(px))
    hues = (rng.uniform(0, 180, len(px)) + t * 60) % 180
    cols = cv2.cvtColor(np.dstack([hues, np.full_like(hues, 90), np.full_like(hues, 255)])
                        .astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    for x, y, r, c in zip(px, py, sizes, cols):
        c = tuple(int(v) for v in c)
        r = int(r)
        cv2.line(sparkle, (x - r, y), (x + r, y), c, 1, cv2.LINE_AA)
        cv2.line(sparkle, (x, y - r), (x, y + r), c, 1, cv2.LINE_AA)
        cv2.circle(sparkle, (x, y), 1, (255, 255, 255), -1, cv2.LINE_AA)
    sparkle = cv2.add(sparkle, cv2.GaussianBlur(sparkle, (0, 0), 2.5))
    return cv2.add(base, sparkle)


def f_flowlines(img):
    t = now()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (0, 0), 5).astype(np.float32) / 255.0
    phase = g * 2 * np.pi * 9 - t * 2.2
    lines = (0.5 + 0.5 * np.cos(phase)) ** 10  # thin bright contours
    base = ethereal_grade(img, keep_color=0.1).astype(np.float32) * 0.55
    copper = np.array([70, 150, 255], np.float32)
    teal = np.array([230, 200, 60], np.float32)
    tone = g[..., None]
    line_col = copper * tone + teal * (1 - tone)
    glow = cv2.GaussianBlur(lines, (0, 0), 3)[..., None]
    out = base + lines[..., None] * line_col * 1.1 + glow * line_col * 0.8
    return np.clip(out, 0, 255).astype(np.uint8)


def f_wisps(img):
    t = now()
    h, w = img.shape[:2]
    xs, ys = _grid(h, w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 100).astype(np.float32)
    mx = xs + 9 * np.sin(ys / 17 + t * 3.1) + 4 * np.sin(ys / 6 - t * 5)
    my = ys + 14 + 6 * np.sin(xs / 23 + t * 2)  # drift upwards
    trail = edges
    acc = edges.copy()
    for i in range(4):  # several smeared copies = wispy streaks
        trail = cv2.remap(trail, mx, my, cv2.INTER_LINEAR) * 0.8
        acc = np.maximum(acc, trail)
    col = _iridescent(h, w, t, speed=50, scale=0.35).astype(np.float32)
    col = col * 0.55 + np.array([60, 150, 240], np.float32) * 0.45  # lean copper
    light = acc[..., None] / 255 * col
    glow = cv2.GaussianBlur(light, (0, 0), 6) * 1.8
    base = ethereal_grade(img).astype(np.float32) * 0.7
    return np.clip(base + light + glow, 0, 255).astype(np.uint8)


# rainbow wings
_trail = {"buf": None, "frame_id": -1, "prev": None}
_frame_ctx = {"frame": None, "offset": (0, 0), "frame_id": 0}


def f_wings(img):
    """Moving parts leave rainbow light trails, like feathered wings."""
    t = now()
    full = _frame_ctx["frame"]
    h, w = img.shape[:2]
    base = ethereal_grade(img)
    if full is None:
        return base
    H, W = full.shape[:2]
    sh, sw = H // 2, W // 2
    if _trail["frame_id"] != _frame_ctx["frame_id"]:  # update once per frame
        small = cv2.resize(full, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        hue = np.full((sh, sw), (t * 140) % 180)
        tint = _hsv_to_bgr(hue, np.full((sh, sw), 230), np.full((sh, sw), 255)).astype(np.float32)
        prev = _trail["prev"]
        motion = np.zeros_like(gray) if prev is None or prev.shape != gray.shape else np.abs(gray - prev)
        _trail["prev"] = gray
        motion = cv2.GaussianBlur(motion, (0, 0), 1.5)
        cur = tint * np.clip(motion * 3 + gray ** 4 * 0.2, 0, 1)[..., None]
        buf = _trail["buf"]
        if buf is None or buf.shape != cur.shape:
            buf = cur
        else:
            buf = np.maximum(buf * 0.92, cur)
        _trail["buf"], _trail["frame_id"] = buf, _frame_ctx["frame_id"]
    x0, y0 = _frame_ctx["offset"]
    big = cv2.resize(_trail["buf"], (W, H), interpolation=cv2.INTER_LINEAR)[y0:y0 + h, x0:x0 + w]
    if big.shape[:2] != (h, w):
        return base
    streak = cv2.GaussianBlur(big, (0, 0), 2)
    return _bloom(_screen(base, np.clip(streak, 0, 255)), 10, 0.5, 150)


FILTERS = [
    ("Ethereal", f_ethereal),
    ("Halo", f_halo),
    ("Light beam", f_beam),
    ("Liquid chrome", f_chrome),
    ("Glitter", f_glitter),
    ("Flow lines", f_flowlines),
    ("Wisps", f_wisps),
    ("Rainbow wings", f_wings),
]


# hands geometry
def to_px(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def hand_size(hand, w, h):
    return float(np.linalg.norm(to_px(hand[WRIST], w, h) - to_px(hand[MIDDLE_MCP], w, h))) + 1e-6


def pinky_thumb_touch(hand, w, h):
    s = hand_size(hand, w, h)
    thumb = to_px(hand[THUMB_TIP], w, h)
    pinky = to_px(hand[PINKY_TIP], w, h)
    index = to_px(hand[INDEX_TIP], w, h)
    touching = np.linalg.norm(pinky - thumb) < 0.35 * s
    index_away = np.linalg.norm(index - thumb) > 0.6 * s
    return touching and index_away, (thumb + pinky) / 2


def _segments_cross(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def portal_points(hands, w, h):
    if len(hands) < 2:
        return None
    a, b = sorted(hands[:2], key=lambda hd: hd[WRIST].x)  # left/right on screen
    pts = np.array([to_px(a[INDEX_TIP], w, h), to_px(b[INDEX_TIP], w, h),
                    to_px(b[THUMB_TIP], w, h), to_px(a[THUMB_TIP], w, h)], np.float32)
    return pts


def polygon_for_drawing(pts):
    p = pts.astype(np.int32)
    if _segments_cross(p[0], p[1], p[2], p[3]) or _segments_cross(p[1], p[2], p[3], p[0]):
        return cv2.convexHull(p).reshape(-1, 2)
    return p


class Smoother:

    def __init__(self, alpha=0.55, hold_frames=5):
        self.alpha, self.hold_frames = alpha, hold_frames
        self.pts, self.missing = None, 0

    def update(self, pts):
        if pts is None:
            self.missing += 1
            if self.missing > self.hold_frames:
                self.pts = None
            return self.pts
        self.missing = 0
        self.pts = pts.copy() if self.pts is None else self.alpha * pts + (1 - self.alpha) * self.pts
        return self.pts


# render
def apply_in_polygon(frame, poly, filt, feather=2):
    x, y, bw, bh = cv2.boundingRect(poly)
    H, W = frame.shape[:2]
    x0, y0 = max(x - 8, 0), max(y - 8, 0)
    x1, y1 = min(x + bw + 8, W), min(y + bh + 8, H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 255, lineType=cv2.LINE_AA)
    if not mask.any():
        return
    if feather:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    _frame_ctx["offset"] = (x0, y0)
    filtered = filt(roi.copy()).astype(np.float32)
    frame[y0:y1, x0:x1] = (filtered * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)


def split_polygon(poly):
    """Split along the diagonal from point 0 to point 2 (left index -> right thumb)."""
    if len(poly) < 4:
        return poly, None
    return poly[[0, 1, 2]], poly[[2, 3, 0]]


def draw_glow_border(frame, poly):
    """Rainbow glowing outline that cycles colour."""
    t = now()
    x, y, bw, bh = cv2.boundingRect(poly)
    H, W = frame.shape[:2]
    pad = 30
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, W), min(y + bh + pad, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    p = poly - [x0, y0]
    n = len(p)
    for i in range(n):
        hue = int((t * 90 + i * 180 / n) % 180)
        col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.line(layer, tuple(map(int, p[i])), tuple(map(int, p[(i + 1) % n])), col, 6, cv2.LINE_AA)
    glow = cv2.GaussianBlur(layer, (0, 0), 9)
    roi = frame[y0:y1, x0:x1]
    roi[:] = cv2.add(roi, cv2.convertScaleAbs(glow, alpha=1.8))
    cv2.polylines(roi, [p.astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_text(frame, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    x0, y0 = max(x - 6, 0), max(y - th - 6, 0)
    x1, y1 = min(x + tw + 6, frame.shape[1]), min(y + base + 4, frame.shape[0])
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = (frame[y0:y1, x0:x1] * 0.35).astype(np.uint8)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame, idx, split, show_help, flash, cam_index):
    name = FILTERS[idx][0]
    if split:
        name += "  +  " + FILTERS[(idx + 1) % len(FILTERS)][0]
    draw_text(frame, name, (18, 40), 0.9, (80, 255, 255) if flash else (255, 255, 255), 2)
    if show_help:
        H = frame.shape[0]
        lines = ["Frame: both index fingers + both thumbs",
                 "Pinky touches thumb / n: next filter   p: previous",
                 f"c: switch camera (now {cam_index})   s: split   m: mirror   h: hide   q: quit"]
        for k, t in enumerate(lines):
            draw_text(frame, t, (14, H - 18 - (len(lines) - 1 - k) * 24), 0.52)


# camera
def load_settings():
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(s, f)
    except Exception:
        pass


def camera_backend():
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def mac_camera_names():
    """Camera names macOS reports (None if unknown / not on a Mac)."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                             capture_output=True, text=True, timeout=10).stdout
        return [c.get("_name", "?") for c in json.loads(out).get("SPCameraDataType", [])]
    except Exception:
        return None


def open_camera(index, width, height, warmup_s=4.0):
    cap = cv2.VideoCapture(index, camera_backend())
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    deadline = time.monotonic() + warmup_s
    while time.monotonic() < deadline:
        ok, fr = cap.read()
        if ok and fr is not None and fr.size:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


def next_camera(current, width, height, count):
    """Try every other camera index, then fall back to the current one."""
    for step in range(1, count):
        idx = (current + step) % count
        cap = open_camera(idx, width, height)
        if cap is not None:
            return idx, cap
    return current, open_camera(current, width, height)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~7 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def main():
    ap = argparse.ArgumentParser(description="Hand-gesture filter portal")
    ap.add_argument("--camera", type=int, default=None,
                    help="webcam index (default: last one you picked with 'c', else 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-mirror", action="store_true", help="don't flip the image horizontally")
    args = ap.parse_args()

    settings = load_settings()
    names = mac_camera_names()
    if names is not None:
        print("Cameras found:", ", ".join(f"{i}: {n}" for i, n in enumerate(names)) or "none")
        if not names:
            sys.exit("macOS reports no camera. If your MacBook lid is closed, open it, "
                     "or connect a webcam / iPhone.")
    count = len(names) if names else 4
    cam_index = args.camera if args.camera is not None else settings.get("camera", 0)
    if cam_index >= count:
        cam_index = 0
    cap = open_camera(cam_index, args.width, args.height)
    if cap is None:
        cam_index, cap = next_camera(cam_index, args.width, args.height, count)
    if cap is None:
        sys.exit("Could not get an image from the camera. On macOS: System Settings > "
                 "Privacy & Security > Camera, turn on your terminal app, then quit and "
                 "reopen the terminal. Also close other apps using the camera (FaceTime, Zoom...).")
    print(f"Using camera {cam_index}. Press 'c' in the window to switch camera.")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    idx, split, mirror, show_help = 0, False, not args.no_mirror, True
    smoother = Smoother()
    last_switch, gesture_armed = 0.0, True
    t0, last_ts = now(), -1
    win = "Filter Portal"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("The camera returned no image.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            ts = max(int((now() - t0) * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            hands = result.hand_landmarks or []

            # Gesture: pinky touches thumb -> next filter
            touch_points = []
            for hd in hands:
                touching, where = pinky_thumb_touch(hd, W, H)
                if touching:
                    touch_points.append(where)
            t = now()
            if touch_points and gesture_armed and t - last_switch > 0.8:
                idx = (idx + 1) % len(FILTERS)
                last_switch, gesture_armed = t, False
            elif not touch_points:
                gesture_armed = True  # separate the fingers before switching again

            _frame_ctx["frame"] = frame.copy()
            _frame_ctx["frame_id"] += 1
            pts = smoother.update(portal_points(hands, W, H))
            if pts is not None:
                poly = polygon_for_drawing(pts)
                if cv2.contourArea(poly) > 600:
                    if split:
                        a, b = split_polygon(poly)
                        apply_in_polygon(frame, a, FILTERS[idx][1])
                        if b is not None:
                            apply_in_polygon(frame, b, FILTERS[(idx + 1) % len(FILTERS)][1])
                    else:
                        apply_in_polygon(frame, poly, FILTERS[idx][1])
                    draw_glow_border(frame, poly)

            for p in touch_points:
                cv2.circle(frame, tuple(int(v) for v in p), 14, (80, 255, 255), 2, cv2.LINE_AA)

            draw_hud(frame, idx, split, show_help, t - last_switch < 0.5, cam_index)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("n"), ord(" ")):
                idx, last_switch = (idx + 1) % len(FILTERS), t
            elif key == ord("p"):
                idx, last_switch = (idx - 1) % len(FILTERS), t
            elif key == ord("s"):
                split = not split
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("c"):
                if count < 2:
                    print("Only one camera available:", names[0] if names else cam_index)
                else:
                    cap.release()
                    cam_index, cap = next_camera(cam_index, args.width, args.height, count)
                    if cap is None:
                        sys.exit("Lost the camera while switching.")
                    settings["camera"] = cam_index
                    save_settings(settings)
                    label = names[cam_index] if names and cam_index < len(names) else cam_index
                    print(f"Switched to camera {label} (saved).")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
