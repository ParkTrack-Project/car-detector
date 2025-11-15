#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Straighten wide-angle image by clicking points that are collinear in the real world.
Now with guaranteed no-crop output: whole original is visible; outside is black.

Controls:
- Left Click: add point to current line
- n: start a new line
- u: undo last point
- r: reset all points
- ENTER: fit distortion, undistort and save (if --out-*)
- q / ESC: quit
"""

import argparse
import json
import sys
from pathlib import Path
import time
import random

import cv2
import numpy as np

# ---------- I/O helpers ----------

def read_first_frame(src: str, timeout_sec: float = 3.0):
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    frame = None
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        ok, fr = cap.read()
        if ok and fr is not None:
            frame = fr
            break
        time.sleep(0.05)
    cap.release()
    return frame

def save_calib(path: Path, w, h, K, D, balance=0.0):
    data = {
        "image_width": int(w),
        "image_height": int(h),
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "balance": float(balance),
        "model": "opencv_fisheye_k1k2k3k4"
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# ---------- Geometry / model ----------

def K_from_params(w, h, fx_scale=0.9, fy_scale=0.9, cx_off=0.0, cy_off=0.0):
    fx = float(fx_scale) * w
    fy = float(fy_scale) * w
    cx = w / 2.0 + float(cx_off)
    cy = h / 2.0 + float(cy_off)
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def undistort_points_fisheye(pts_px, K, D, P=None):
    """pts_px: (N,2) pixel coords -> undistorted coords (pixel if P, normalized if P=None)."""
    if len(pts_px) == 0:
        return np.zeros((0,2), np.float64)
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1,1,2)
    und = cv2.fisheye.undistortPoints(pts, K, D, P=P).reshape(-1,2)
    return und

def line_fit_residuals(pts):
    """Sum of squared perpendicular distances to best-fit line."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    c = pts.mean(axis=0, keepdims=True)
    A = pts - c
    _, _, vh = np.linalg.svd(A)
    normal = vh[1]
    d = np.abs((A @ normal.reshape(2,1))).ravel()
    return float(np.sum(d**2))

def total_collinearity_cost(groups, K, D):
    if len(groups) == 0:
        return 1e18
    cost = 0.0
    for g in groups:
        if len(g) < 2:
            continue
        und = undistort_points_fisheye(g, K, D, P=None)  # normalized
        cost += line_fit_residuals(und)
    return cost

# ---------- Search (grid + stochastic refine) ----------

def fit_params_by_collinearity(groups, w, h, seed=0):
    rng = random.Random(seed)

    # coarse grid for k1,k2
    grid_k = np.linspace(-0.45, 0.20, 14)
    best = None
    best_tuple = None

    fx0 = 0.9; fy0 = 0.9
    cx0 = 0.0; cy0 = 0.0
    for k1 in grid_k:
        for k2 in grid_k:
            K = K_from_params(w, h, fx0, fy0, cx0, cy0)
            D = np.array([[k1],[k2],[0.0],[0.0]], dtype=np.float64)
            c = total_collinearity_cost(groups, K, D)
            if (best is None) or (c < best):
                best = c
                best_tuple = (fx0, fy0, cx0, cy0, k1, k2, 0.0, 0.0)

    fx, fy, cx, cy, k1, k2, k3, k4 = best_tuple

    # local random refine
    s_fx, s_fy = 0.10, 0.10
    s_cx, s_cy = 0.04 * w, 0.04 * h
    s_k1, s_k2, s_k3, s_k4 = 0.05, 0.05, 0.01, 0.01

    def clamp(v, lo, hi): return max(lo, min(hi, v))

    K = K_from_params(w, h, fx, fy, cx, cy)
    D = np.array([[k1],[k2],[k3],[k4]], dtype=np.float64)
    base_cost = total_collinearity_cost(groups, K, D)

    for it in range(300):
        fx_p = clamp(rng.gauss(fx, s_fx), 0.4, 2.0)
        fy_p = clamp(rng.gauss(fy, s_fy), 0.4, 2.0)
        cx_p = clamp(rng.gauss(cx, s_cx), -0.2*w, 0.2*w)
        cy_p = clamp(rng.gauss(cy, s_cy), -0.2*h, 0.2*h)
        k1_p = clamp(rng.gauss(k1, s_k1), -0.8, 0.8)
        k2_p = clamp(rng.gauss(k2, s_k2), -0.8, 0.8)
        k3_p = clamp(rng.gauss(k3, s_k3), -0.8, 0.8)
        k4_p = clamp(rng.gauss(k4, s_k4), -0.8, 0.8)

        Kp = K_from_params(w, h, fx_p, fy_p, cx_p, cy_p)
        Dp = np.array([[k1_p],[k2_p],[k3_p],[k4_p]], dtype=np.float64)
        c = total_collinearity_cost(groups, Kp, Dp)
        if c < base_cost:
            fx, fy, cx, cy, k1, k2, k3, k4 = fx_p, fy_p, cx_p, cy_p, k1_p, k2_p, k3_p, k4_p
            K, D = Kp, Dp
            base_cost = c

        if (it+1) % 60 == 0:
            s_fx *= 0.6; s_fy *= 0.6
            s_cx *= 0.6; s_cy *= 0.6
            s_k1 *= 0.6; s_k2 *= 0.6; s_k3 *= 0.6; s_k4 *= 0.6

    return K, D, base_cost

# ---------- No-crop maps ----------

def build_fullview_maps_fisheye(w, h, K, D, margin_px=0):
    """
    Создаём newK, чтобы вся исходная картинка гарантированно уместилась в выход (w x h).
    Делаем это аналитически на границе изображения: считаем выпрямленные координаты
    для точек по периметру (нормализованные), подбираем scale и сдвиг.
    """
    # 1) соберём точки по периметру (достаточно шаг 8–16 px)
    step = max(8, int(min(w, h) / 160))
    xs = np.arange(0, w, step, dtype=np.float64)
    ys = np.arange(0, h, step, dtype=np.float64)
    border_pts = []
    for x in xs:
        border_pts.append([x, 0.0])
        border_pts.append([x, h-1.0])
    for y in ys:
        border_pts.append([0.0, y])
        border_pts.append([w-1.0, y])
    border_pts = np.array(border_pts, dtype=np.float64)

    # 2) выпрямим в НОРМАЛИЗОВАННЫЕ координаты (P=None)
    und_norm = undistort_points_fisheye(border_pts, K, D, P=None)  # (N,2)
    min_x, min_y = und_norm[:,0].min(), und_norm[:,1].min()
    max_x, max_y = und_norm[:,0].max(), und_norm[:,1].max()

    # 3) масштаб и сдвиг: хотим, чтобы [min,max] попал внутрь [margin, w-margin]
    avail_w = (w - 1 - 2*margin_px)
    avail_h = (h - 1 - 2*margin_px)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 0 or span_y <= 0:
        span_x = max(span_x, 1e-6)
        span_y = max(span_y, 1e-6)

    sx = avail_w / span_x
    sy = avail_h / span_y
    s = min(sx, sy)  # одинаковый масштаб по X/Y, чтобы сохранить пропорции

    cx = margin_px - min_x * s
    cy = margin_px - min_y * s

    newK = np.array([[s, 0.0, cx],
                     [0.0, s, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

    # 4) карты для ремапа
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), newK, (w, h), cv2.CV_16SC2
    )
    return map1, map2, newK

# ---------- Click UI ----------

class ClickUI:
    def __init__(self, img):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.groups = [[]]   # list of lines; active is last
        self.win = "Click collinear points (ENTER=solve, n=new line, u=undo, r=reset, q=quit)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def on_mouse(self, evt, x, y, flags, param=None):
        if evt == cv2.EVENT_LBUTTONDOWN:
            self.groups[-1].append([float(x), float(y)])

    def undo(self):
        if len(self.groups) == 0: return
        if len(self.groups[-1]) > 0:
            self.groups[-1].pop()
        else:
            if len(self.groups) > 1:
                self.groups.pop()

    def new_line(self):
        if len(self.groups[-1]) == 0:
            return
        self.groups.append([])

    def reset(self):
        self.groups = [[]]

    def draw(self):
        vis = self.img.copy()
        colors = [(0,200,255),(0,255,0),(255,0,0),(180,0,180),(0,140,255),(0,255,255)]
        for i, g in enumerate(self.groups):
            if len(g) == 0: continue
            col = colors[i % len(colors)]
            for p in g:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, col, -1, lineType=cv2.LINE_AA)
            if len(g) >= 2:
                for j in range(len(g)-1):
                    p1 = (int(g[j][0]), int(g[j][1]))
                    p2 = (int(g[j+1][0]), int(g[j+1][1]))
                    cv2.line(vis, p1, p2, col, 2, lineType=cv2.LINE_AA)
        msg1 = "LMB:add | n:new line | u:undo | r:reset | ENTER:solve | q:quit"
        cv2.putText(vis, msg1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, msg1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        return vis

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Straighten fisheye image by clicking collinear points (no-crop output)")
    ap.add_argument("--source", required=True, help="Image path or video/URL")
    ap.add_argument("--out-img", type=str, default="", help="Save undistorted result here")
    ap.add_argument("--out-calib", type=str, default="", help="Save fitted calibration JSON here")
    ap.add_argument("--margin", type=int, default=0, help="Extra black margin around (px)")
    ap.add_argument("--show", action="store_true", help="Show result windows")
    args = ap.parse_args()

    frame = read_first_frame(args.source)
    if frame is None:
        print(f"[ERR] cannot read frame from {args.source}", file=sys.stderr); sys.exit(2)
    h, w = frame.shape[:2]

    ui = ClickUI(frame)
    while True:
        vis = ui.draw()
        cv2.imshow(ui.win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            cv2.destroyAllWindows()
            sys.exit(0)
        if key in (ord('n'),):
            ui.new_line()
        if key in (ord('u'),):
            ui.undo()
        if key in (ord('r'),):
            ui.reset()
        if key == 13:  # ENTER
            groups = [np.array(g, dtype=np.float64) for g in ui.groups if len(g) >= 2]
            if len(groups) == 0:
                print("[WARN] Need >= 2 points per line (at least one line).", file=sys.stderr)
                continue

            print(f"[INFO] Fitting on {sum(len(g) for g in groups)} points across {len(groups)} lines...", file=sys.stderr)
            K, D, cost = fit_params_by_collinearity(groups, w, h, seed=0)
            print(f"[INFO] Best cost: {cost:.3f}", file=sys.stderr)

            # no-crop maps: whole image must fit into output size
            map1, map2, newK = build_fullview_maps_fisheye(w, h, K, D, margin_px=args.margin)
            rect = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            if args.out_calib:
                outc = Path(args.out_calib); outc.parent.mkdir(parents=True, exist_ok=True)
                save_calib(outc, w, h, K, D, balance=0.0)
                print(f"[OK] calib saved to {outc}", file=sys.stderr)
            if args.out_img:
                outi = Path(args.out_img); outi.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(outi), rect)
                print(f"[OK] rectified image saved to {outi}", file=sys.stderr)

            if args.show:
                before = frame.copy()
                for g in groups:
                    for p in g:
                        cv2.circle(before, (int(p[0]), int(p[1])), 3, (0,255,255), -1)
                after = rect.copy()
                for g in groups:
                    und = undistort_points_fisheye(g, K, D, P=newK)
                    for p in und:
                        cv2.circle(after, (int(round(p[0])), int(round(p[1]))), 3, (0,255,0), -1)
                cv2.imshow("before", before)
                cv2.imshow("after (no-crop rectified)", after)
                cv2.waitKey(0)
                cv2.destroyWindow("before")
                cv2.destroyWindow("after (no-crop rectified)")
            # остаёмся в режиме разметки для донастройки

if __name__ == "__main__":
    main()
