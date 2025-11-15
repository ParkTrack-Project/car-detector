#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO(OpenVINO) + ИЗОГНУТЫЕ парковочные зоны на ИСКАЖЁННОМ кадре.
Цвет подписей совпадает с цветом зоны; добавлена легенда в левом нижнем углу.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import yaml
import numpy as np
from openvino import Core, Tensor

# ------------------------
# OpenVINO / YOLO utils
# ------------------------

def load_class_names(model_dir: Path):
    meta = model_dir / "metadata.yaml"
    if meta.exists():
        try:
            data = yaml.safe_load(meta.read_text(encoding="utf-8"))
            names = data.get("names") or data.get("classes") or None
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
            if isinstance(names, list) and len(names) > 0:
                return names
        except Exception:
            pass
    return ["car"]

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh_to_xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_thres=0.5):
    if boxes is None or len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    w = np.maximum(0.0, x2 - x1); h = np.maximum(0.0, y2 - y1)
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        idxs = order[1:]
        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[idxs] - inter + 1e-9)
        remaining = np.where(iou <= iou_thres)[0]
        order = order[remaining + 1]
    return keep

def draw_box_with_alpha(frame, box, label_text, edge_color, fill_color=None, alpha=0.25, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
    if fill_color is not None and x2 > x1 and y2 > y1:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), edge_color, thickness)
    if label_text:
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # аутлайн
        cv2.putText(frame, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, edge_color, 1, cv2.LINE_AA)

def parse_with_embedded_nms(output_tensors, conf_thres):
    for t in output_tensors:
        arr = t.data if hasattr(t, "data") else np.asarray(t)
        a = np.array(arr)
        if a.ndim == 3 and a.shape[-1] == 6:
            det = a[0]
            if det.size == 0:
                return np.empty((0,4)), np.empty((0,)), np.empty((0,))
            boxes = det[:, :4]; scores = det[:, 4]; cls_ids = det[:, 5]
            m = scores >= conf_thres
            return boxes[m], scores[m], cls_ids[m]
    return None

def parse_raw_yolo_outputs(output_tensors, conf_thres, car_only=False, car_id=0, nms_iou=0.5):
    arrays = []
    for t in output_tensors:
        arr = t.data if hasattr(t, "data") else np.asarray(t)
        arrays.append(np.array(arr))
    pred = max(arrays, key=lambda a: a.size)

    if pred.ndim == 3:
        if pred.shape[1] >= 5 and pred.shape[0] == 1:
            pred = np.transpose(pred[0], (1, 0))
        elif pred.shape[2] >= 5 and pred.shape[0] == 1:
            pred = pred[0]
        elif pred.shape[0] == 1 and pred.shape[1] == 1:
            pred = pred[0,0]
    elif pred.ndim != 2:
        raise RuntimeError("Неизвестный формат выхода модели.")

    if pred.shape[1] < 5:
        raise RuntimeError("Выход модели не похож на YOLO-предсказания.")

    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    m = scores >= conf_thres
    boxes_xywh = boxes_xywh[m]; scores = scores[m]; cls_ids = cls_ids[m]

    if car_only:
        car_mask = (cls_ids == car_id)
        boxes_xywh = boxes_xywh[car_mask]; scores = scores[car_mask]; cls_ids = cls_ids[car_mask]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, iou_thres=nms_iou)
    if len(keep) == 0:
        return np.empty((0,4)), np.empty((0,)), np.empty((0,))
    keep = np.array(keep, dtype=int)
    return boxes_xyxy[keep], scores[keep], cls_ids[keep]

# ------------------------
# Calibration I/O
# ------------------------

def load_calib(calib_path: Path):
    data = json.loads(calib_path.read_text(encoding="utf-8"))
    w = int(data["image_width"]); h = int(data["image_height"])
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64).reshape(-1,1)
    newK = np.array(data["newK"], dtype=np.float64) if "newK" in data else None
    balance = float(data.get("balance", 0.0))
    return (w, h, K, D, newK, balance)

def compute_newK_fullview(w, h, K, D):
    step = max(8, int(min(w, h) / 160))
    xs = np.arange(0, w, step, dtype=np.float64)
    ys = np.arange(0, h, step, dtype=np.float64)
    border = []
    for x in xs:
        border += [[x,0.0],[x,h-1.0]]
    for y in ys:
        border += [[0.0,y],[w-1.0,y]]
    border = np.array(border, dtype=np.float64)
    und = cv2.fisheye.undistortPoints(border.reshape(-1,1,2), K, D, P=None).reshape(-1,2)
    min_x, min_y = und[:,0].min(), und[:,1].min()
    max_x, max_y = und[:,0].max(), und[:,1].max()
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    s = min((w-1)/span_x, (h-1)/span_y)
    cx = -min_x*s
    cy = -min_y*s
    newK = np.array([[s,0.0,cx],[0.0,s,cy],[0.0,0.0,1.0]], dtype=np.float64)
    return newK

# ------------------------
# Curved zones from 4 anchors
# ------------------------

def load_zones_orig(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    zones_raw = data.get("zones") or []
    zones = [np.array([[p["x"], p["y"]] for p in z], dtype=np.float64) for z in zones_raw]
    return zones, data.get("image_width"), data.get("image_height")

def undistort_pixels_to_rectified(p_d, K, D, newK):
    pts = np.asarray(p_d, dtype=np.float64).reshape(-1,1,2)
    pr = cv2.fisheye.undistortPoints(pts, K, D, P=newK).reshape(-1,2)
    return pr

def rectified_to_distorted_pixels(p_r, K, D, newK):
    pr = np.asarray(p_r, dtype=np.float64).reshape(-1,2)
    ones = np.ones((pr.shape[0],1), dtype=np.float64)
    homo = np.hstack([pr, ones])
    inv_newK = np.linalg.inv(newK)
    norm = (inv_newK @ homo.T).T[:, :2].reshape(-1,1,2)
    pd = cv2.fisheye.distortPoints(norm, K, D).reshape(-1,2)
    return pd

def densify_edge(p0, p1, n_samples):
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64).reshape(-1,1)
    pts = (1.0 - t) * p0 + t * p1
    return pts

def build_curved_polygon_from_anchors(anchors_px_distorted, K, D, newK, samples_per_edge=50):
    a = np.asarray(anchors_px_distorted, dtype=np.float64)
    pr = undistort_pixels_to_rectified(a, K, D, newK)
    pts_curve = []
    for i in range(len(pr)):
        p0 = pr[i]
        p1 = pr[(i+1) % len(pr)]
        seg = densify_edge(p0, p1, samples_per_edge)
        seg_d = rectified_to_distorted_pixels(seg, K, D, newK)
        pts_curve.append(seg_d)
    poly = np.vstack(pts_curve).astype(np.float32)
    return poly

def point_in_poly_cv(point_xy, polygon_float32):
    cnt = polygon_float32.reshape((-1,1,2)).astype(np.float32)
    res = cv2.pointPolygonTest(cnt, (float(point_xy[0]), float(point_xy[1])), False)
    return res >= 0

def assign_zone_for_box(box, curved_polys, h_img, w_img, mode, w_center=1.0, w_overlap=2.0):
    """
    Возвращает (best_idx, depth_px, overlap_ratio, best_score).
    mode: "center" | "depth" | "hybrid"
    """
    x1, y1, x2, y2 = box
    cx = float((x1 + x2) * 0.5)
    cy = float((y1 + y2) * 0.5)
    center = (cx, cy)

    best_idx = -1
    best_score = -1.0
    best_depth = 0.0
    best_overlap = 0.0

    for zi, poly in enumerate(curved_polys):
        if mode == "center":
            depth_px = signed_depth_to_polygon(center, poly)
            score = depth_px
            overlap = 0.0
        elif mode == "depth":
            depth_px = signed_depth_to_polygon(center, poly)
            score = depth_px
            overlap = 0.0
        else:  # hybrid
            depth_px = signed_depth_to_polygon(center, poly)
            depth_norm = norm_depth_for_box(depth_px, box)
            overlap = overlap_ratio_box_in_polygon(box, poly, h_img, w_img)
            score = w_center * depth_norm + w_overlap * overlap

        if score > best_score:
            best_score = score
            best_idx = zi
            best_depth = depth_px
            best_overlap = overlap

    if best_score <= 0.0:
        return -1, 0.0, 0.0, 0.0
    return best_idx, best_depth, best_overlap, best_score


# ------------------------
# Viz helpers
# ------------------------

def vivid_palette(n):
    base = [
        (0, 180, 255), (0, 200, 0), (255, 0, 0), (180, 0, 180),
        (0, 140, 255), (0, 255, 255), (255, 0, 180), (255, 255, 0),
        (0, 100, 255), (255, 100, 0),
    ]
    return [base[i % len(base)] for i in range(n)]

def draw_polygon_outline(frame, polygon, color_bgr, thickness=2):
    pts = polygon.reshape((-1,1,2)).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color_bgr, thickness=thickness)

def put_text_outline(frame, text, org, color_bgr, scale=0.6, thick=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thick, cv2.LINE_AA)

def draw_legend_bottom_left(frame, entries, colors, margin=10, padding=8, line_h=22, alpha=0.5):
    """
    entries: list[str], colors: list[(B,G,R)]
    Рисует полупрозрачный чёрный бокс и строки легенды в левой нижней части кадра.
    """
    h, w = frame.shape[:2]
    # вычислим ширину
    widths = []
    for t in entries:
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        widths.append(tw)
    box_w = max(widths) + 2*padding
    box_h = len(entries)*line_h + 2*padding

    x0 = margin
    y1 = h - margin
    y0 = y1 - box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+box_w, y1), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, dst=frame)

    # строки
    y = y0 + padding + int(line_h*0.8)
    for i, t in enumerate(entries):
        put_text_outline(frame, t, (x0+padding, y), colors[i], scale=0.6, thick=1)
        y += line_h

def signed_depth_to_polygon(point_xy, polygon_float32):
    """Положительная глубина (в пикселях) внутри полигона, 0 снаружи."""
    cnt = polygon_float32.reshape((-1,1,2)).astype(np.float32)
    d = cv2.pointPolygonTest(cnt, (float(point_xy[0]), float(point_xy[1])), True)
    return max(0.0, float(d))  # снаружи = 0

def overlap_ratio_box_in_polygon(box_xyxy, polygon_float32, full_h, full_w):
    """
    Доля площади прямоугольника, попавшая внутрь полигона ∈ [0..1].
    Считаем аккуратно через растеризацию только в ограниченном ROI.
    """
    x1, y1, x2, y2 = box_xyxy
    x1i = int(np.floor(max(0, min(x1, x2))))
    x2i = int(np.ceil (min(full_w-1, max(x1, x2))))
    y1i = int(np.floor(max(0, min(y1, y2))))
    y2i = int(np.ceil (min(full_h-1, max(y1, y2))))
    if x2i <= x1i or y2i <= y1i:
        return 0.0

    # ограничим полигон и бокс ROI
    roi_w, roi_h = x2i - x1i + 1, y2i - y1i + 1
    poly_roi = polygon_float32.copy()
    poly_roi[:,0] -= x1i
    poly_roi[:,1] -= y1i

    mask_poly = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(mask_poly, [poly_roi.astype(np.int32)], 255)

    mask_box = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.rectangle(mask_box, (int(round(x1 - x1i)), int(round(y1 - y1i))),
                              (int(round(x2 - x1i)), int(round(y2 - y1i))), 255, -1)

    inter = cv2.countNonZero(cv2.bitwise_and(mask_poly, mask_box))
    box_area = cv2.countNonZero(mask_box)
    if box_area <= 0:
        return 0.0
    return float(inter) / float(box_area)

def norm_depth_for_box(depth_px, box_xyxy):
    """Нормируем глубину на характерный размер бокса (чтобы быть сопоставимой с overlap)."""
    w = max(1.0, float(box_xyxy[2] - box_xyxy[0]))
    h = max(1.0, float(box_xyxy[3] - box_xyxy[1]))
    diag = (w*w + h*h)**0.5
    return float(depth_px) / diag  # ~[0..~0.5]


# ------------------------
# Main (ONE FRAME)
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="YOLO + Curved parking zones on distorted image (one frame)")
    ap.add_argument("--model", required=True, help="OpenVINO model.xml")
    ap.add_argument("--source", required=True, help="URL/Path to video/image")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="AUTO")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--car_only", action="store_true")
    ap.add_argument("--zones", required=True, help="JSON with zones (anchors on distorted image)")
    ap.add_argument("--calib", required=True, help="Calibration JSON (K,D and optionally newK)")
    ap.add_argument("--samples-per-edge", type=int, default=60, help="Sampling density for curved edges")
    ap.add_argument("--zone_thickness", type=int, default=2)
    ap.add_argument("--car_alpha", type=float, default=0.25)
    ap.add_argument("--out_img", default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--assign", choices=["center", "depth", "hybrid"], default="hybrid",
                    help="Способ выбора зоны: center=первая, глубина центра; depth=макс. глубина; hybrid=глубина+процент площади")
    ap.add_argument("--w_center", type=float, default=1.0, help="Вес глубины центра для hybrid")
    ap.add_argument("--w_overlap", type=float, default=2.0, help="Вес доли площади для hybrid")

    args = ap.parse_args()

    # --- Calibration ---
    calib_path = Path(args.calib).expanduser().resolve()
    if not calib_path.exists():
        print(f"[ERR] calib not found: {calib_path}", file=sys.stderr); sys.exit(1)
    cw, ch, K, D, newK_opt, _ = load_calib(calib_path)

    # --- Read first frame ---
    cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[ERR] cannot open source: {args.source}", file=sys.stderr); sys.exit(2)
    frame = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ok, fr = cap.read()
        if ok and fr is not None:
            frame = fr; break
        time.sleep(0.05)
    cap.release()
    if frame is None:
        print("[ERR] failed to read frame", file=sys.stderr); sys.exit(3)
    h0, w0 = frame.shape[:2]

    if (cw != w0) or (ch != h0):
        sx = w0 / cw; sy = h0 / ch
        K = K.copy()
        K[0,0] *= sx; K[1,1] *= sy
        K[0,2] *= sx; K[1,2] *= sy

    newK = newK_opt if newK_opt is not None else compute_newK_fullview(w0, h0, K, D)

    # --- Zones (anchors on distorted image) -> curved polygons ---
    zones_path = Path(args.zones).expanduser().resolve()
    if not zones_path.exists():
        print(f"[ERR] zones json not found: {zones_path}", file=sys.stderr); sys.exit(1)
    zones_anchors, _, _ = load_zones_orig(zones_path)

    curved_polys = []
    for z in zones_anchors:
        poly = build_curved_polygon_from_anchors(
            z, K, D, newK, samples_per_edge=max(8, args.samples_per_edge)
        )
        curved_polys.append(poly)

    zone_colors = vivid_palette(len(curved_polys))
    zone_ids = [i+1 for i in range(len(curved_polys))]

    # --- OpenVINO inference (one frame) ---
    model_xml = Path(args.model).expanduser().resolve()
    if not model_xml.exists():
        print(f"[ERR] model.xml not found: {model_xml}", file=sys.stderr); sys.exit(1)
    names = load_class_names(model_xml.parent)
    car_id = 0

    ie = Core()
    model = ie.read_model(model=model_xml)
    compiled = ie.compile_model(model=model, device_name=args.device)
    input_tensor = compiled.inputs[0]

    img, r, (dw, dh) = letterbox(frame, new_shape=args.imgsz)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    if input_tensor.element_type.type_name == "f16":
        img = img.astype(np.float16)
    else:
        img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))[None, ...]

    infer_req = compiled.create_infer_request()
    infer_req.set_tensor(compiled.inputs[0], Tensor(img))
    infer_req.infer()
    outputs = [infer_req.get_tensor(o) for o in compiled.outputs]

    parsed = parse_with_embedded_nms(outputs, conf_thres=args.conf)
    if parsed is None:
        boxes, scores, cls_ids = parse_raw_yolo_outputs(
            outputs, conf_thres=args.conf,
            car_only=args.car_only or (len(names)==1 and names[0].lower()=="car"),
            car_id=car_id, nms_iou=0.5
        )
    else:
        boxes, scores, cls_ids = parsed
        if args.car_only or (len(names)==1 and names[0].lower()=="car"):
            m = (cls_ids.astype(int) == car_id)
            boxes, scores, cls_ids = boxes[m], scores[m], cls_ids[m]

    if boxes.shape[0] > 0:
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :4] /= r
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0 - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0 - 1)

    # --- Zone accounting with CURVED polygons (depth/overlap aware) ---
    zone_stats = [{"id": int(zone_ids[i]), "occupied": 0, "cars": []} for i in range(len(curved_polys))]
    car_zone_index = [-1] * boxes.shape[0]

    for i, (box, sc, cid) in enumerate(zip(boxes, scores, cls_ids)):
        cx = float((box[0] + box[2]) * 0.5)
        cy = float((box[1] + box[3]) * 0.5)
        center = (cx, cy)

        best_idx = -1
        best_score = -1.0

        for zi, poly in enumerate(curved_polys):
            if args.assign == "center":
                # старая логика, но выбираем наибольшую глубину, а не первую зону
                depth_px = signed_depth_to_polygon(center, poly)
                score = depth_px
            elif args.assign == "depth":
                depth_px = signed_depth_to_polygon(center, poly)
                score = depth_px
            else:
                # hybrid: глубина центра (нормированная) + доля площади бокса в зоне
                depth_px = signed_depth_to_polygon(center, poly)  # >=0 внутри, 0 снаружи
                depth_norm = norm_depth_for_box(depth_px, box)  # ~[0..]
                overlap = overlap_ratio_box_in_polygon(box, poly, h0, w0)  # [0..1]
                score = args.w_center * depth_norm + args.w_overlap * overlap

            if score > best_score:
                best_score = score
                best_idx = zi

        # если вообще нет вклада (снаружи всех зон и пересечений нет) — оставим без зоны
        if best_score <= 0.0:
            assigned = -1
        else:
            assigned = best_idx

        car_zone_index[i] = assigned
        if assigned >= 0:
            zone_stats[assigned]["occupied"] += 1
            zone_stats[assigned]["cars"].append({
                "det_index": i,
                "center": [cx, cy],
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "score": float(sc),
                "class_id": int(cid),
                "depth_px": float(signed_depth_to_polygon(center, curved_polys[assigned])),
                "overlap_ratio": float(overlap_ratio_box_in_polygon(box, curved_polys[assigned], h0, w0))
            })

    # --- Visualization ---
    for zi, poly in enumerate(curved_polys):
        color = zone_colors[zi]
        draw_polygon_outline(frame, poly, color_bgr=color, thickness=args.zone_thickness)
        # подпись тем же цветом (с аутлайном) около «центра тяжести» дискретной кривой
        cxy = np.mean(poly, axis=0)
        label = f"Zone {zone_ids[zi]}: {zone_stats[zi]['occupied']}"
        tx, ty = int(round(cxy[0])), int(round(cxy[1]))
        put_text_outline(frame, label, (max(0, tx - 40), max(12, ty)), color)

    # боксы машин
    for i, (box, sc, cid) in enumerate(zip(boxes, scores, cls_ids)):
        label_name = names[int(cid)] if 0 <= int(cid) < len(names) else str(int(cid))
        pct = int(round(float(sc) * 100))
        label_text = f"{label_name} {pct}%"
        zi = car_zone_index[i]
        if zi >= 0:
            edge = zone_colors[zi]; fill = zone_colors[zi]
            draw_box_with_alpha(frame, box, label_text, edge_color=edge, fill_color=fill,
                                alpha=args.car_alpha, thickness=2)
        else:
            draw_box_with_alpha(frame, box, label_text, edge_color=(0,255,0),
                                fill_color=None, alpha=0.0, thickness=2)
        cx = int(round((box[0] + box[2]) * 0.5)); cy = int(round((box[1] + box[3]) * 0.5))
        cv2.circle(frame, (cx, cy), 3, (0,0,0), -1); cv2.circle(frame, (cx, cy), 2, (255,255,255), -1)

    # --- Legend (bottom-left) ---
    legend_entries = [f"Zone {z['id']}: {z['occupied']}" for z in zone_stats]
    draw_legend_bottom_left(frame, legend_entries, zone_colors, margin=10, padding=8, line_h=22, alpha=0.5)

    # --- JSON ---
    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": str(args.source),
        "zones": zone_stats,
        "totals": {
            "cars_detected": int(boxes.shape[0]),
            "cars_in_zones": int(sum(z["occupied"] for z in zone_stats))
        },
        "meta": {
            "frame_width": int(w0), "frame_height": int(h0),
            "samples_per_edge": int(max(8, args.samples_per_edge))
        }
    }
    print(json.dumps(out, ensure_ascii=False))

    # --- Save/Show ---
    if args.out_img:
        out_path = Path(args.out_img).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            print(f"[WARN] cannot save image to {out_path}", file=sys.stderr)

    if args.show:
        cv2.imshow("Curved zones on distorted image (single frame)", frame)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
