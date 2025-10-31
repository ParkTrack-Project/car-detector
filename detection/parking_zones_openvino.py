#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Пример:
python parking_zones_openvino.py \
  --model ./best_openvino_model/best.xml \
  --source http://109.236.111.203/mjpg/video.mjpg \
  --zones-json zones.json \
  --duration 10 \
  --imgsz 640 --conf 0.15 --iou 0.5 --show
"""

import argparse, json, time, sys
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import yaml
from openvino import Core, Tensor

# ---------- базовые утилы из твоего скрипта ----------

def load_class_names(model_dir: Path):
    meta = model_dir / "metadata.yaml"
    if meta.exists():
        try:
            data = yaml.safe_load(meta.read_text())
            names = data.get("names") or data.get("classes") or None
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
            if isinstance(names, list) and len(names) > 0:
                return names
        except Exception:
            pass
    return ["car"]

def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if (w,h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top,bottom,left,right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh_to_xyxy(x):
    y = np.empty_like(x)
    y[:,0] = x[:,0] - x[:,2]/2
    y[:,1] = x[:,1] - x[:,3]/2
    y[:,2] = x[:,0] + x[:,2]/2
    y[:,3] = x[:,1] + x[:,3]/2
    return y

def nms(boxes, scores, iou_thres=0.5):
    if len(boxes)==0:
        return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        if order.size==1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter/(areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou<=iou_thres)[0]
        order = order[inds+1]
    return keep

def parse_with_embedded_nms(output_tensors, conf_thres):
    for t in output_tensors:
        a = np.array(t.data if hasattr(t,"data") else t)
        if a.ndim==3 and a.shape[-1]==6:
            det = a[0]
            if det.size==0:
                return np.empty((0,4)), np.empty((0,)), np.empty((0,))
            boxes = det[:,:4]; scores = det[:,4]; cls_ids = det[:,5]
            m = scores >= conf_thres
            return boxes[m], scores[m], cls_ids[m]
    return None

def parse_raw_yolo_outputs(output_tensors, conf_thres, iou_thres, car_only=True, car_id=0):
    arrays=[]
    for t in output_tensors:
        arrays.append(np.array(t.data if hasattr(t,"data") else t))
    pred = max(arrays, key=lambda a:a.size)
    if pred.ndim==3:
        if pred.shape[0]==1 and pred.shape[1] >= 5:
            pred = np.transpose(pred[0], (1,0))
        elif pred.shape[0]==1 and pred.shape[2] >= 5:
            pred = pred[0]
        elif pred.shape[0]==1 and pred.shape[1]==1:
            pred = pred[0,0]
    if pred.ndim!=2 or pred.shape[1]<5:
        raise RuntimeError("Unexpected model output shape")
    boxes_xywh = pred[:,:4]
    cls_scores = pred[:,4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    m = scores >= conf_thres
    boxes_xywh = boxes_xywh[m]; scores = scores[m]; cls_ids = cls_ids[m]
    if car_only:
        car_mask = (cls_ids==car_id)
        boxes_xywh = boxes_xywh[car_mask]; scores = scores[car_mask]; cls_ids = cls_ids[car_mask]
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, iou_thres=iou_thres)
    if not keep: return np.empty((0,4)), np.empty((0,)), np.empty((0,))
    keep = np.array(keep, dtype=int)
    return boxes_xyxy[keep], scores[keep], cls_ids[keep]

# ---------- геометрия зон и рисование ----------

def zones_from_json(path: Path):
    data = json.loads(Path(path).read_text())
    zs = []
    for poly in data["zones"]:
        pts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.float32)
        zs.append(pts)
    return zs

def point_in_poly(cx, cy, poly_pts):
    # poly_pts: np.float32 Nx2
    res = cv2.pointPolygonTest(poly_pts.astype(np.float32), (float(cx), float(cy)), False)
    return res >= 0  # внутри или на границе

def draw_zones(frame, zones, colors=None):
    h,w = frame.shape[:2]
    if colors is None:
        colors = [(0,180,255),(255,180,0),(0,255,180),(180,255,0)]
    for i,poly in enumerate(zones):
        pts = poly.reshape(-1,1,2).astype(np.int32)
        cv2.polylines(frame, [pts], True, colors[i%len(colors)], 2)
        cX = int(np.mean(poly[:,0])); cY = int(np.mean(poly[:,1]))
        cv2.circle(frame, (cX,cY), 3, colors[i%len(colors)], -1)
        cv2.putText(frame, f"Zone {i+1}", (cX+6,cY-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Zone {i+1}", (cX+6,cY-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def draw_boxes(frame, boxes, scores, color=(0,255,0), label="car"):
    for (x1,y1,x2,y2), s in zip(boxes, scores):
        pct = int(round(float(s)*100))
        txt = f"{label} {pct}%"
        x1i,y1i,x2i,y2i = map(lambda v:int(round(v)), [x1,y1,x2,y2])
        cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), color, 2)
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1i, max(0,y1i-th-6)), (x1i+tw+4, y1i), color, -1)
        cv2.putText(frame, txt, (x1i+2, y1i-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

# ---------- простой трекинг по IoU и статичность ----------

def box_iou(a, b):
    # a,b: [x1,y1,x2,y2]
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    w = max(0.0, xx2-xx1); h = max(0.0, yy2-yy1)
    inter = w*h
    if inter<=0: return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter/(area_a+area_b-inter+1e-9)

class Track:
    _next_id = 1
    def __init__(self, box, ts, score):
        self.id = Track._next_id; Track._next_id += 1
        self.box = box.astype(np.float32)
        self.score = float(score)
        self.first_ts = ts
        self.last_ts = ts
        self.hits = 1
        self.miss = 0
        self.motion_buf = deque(maxlen=10)  # средняя |разница пикселей| по ROI
        self.prev_roi_gray = None

    def age(self, now): return now - self.first_ts
    def seen_recently(self, now, grace): return (now - self.last_ts) <= grace

def match_tracks(tracks, det_boxes, iou_thr=0.3):
    # жадный матчинг по IoU
    matches = []
    if len(tracks)==0 or len(det_boxes)==0:
        return matches, list(range(len(det_boxes))), list(range(len(tracks)))
    iou_mat = np.zeros((len(tracks), len(det_boxes)), dtype=np.float32)
    for i,tr in enumerate(tracks):
        for j,db in enumerate(det_boxes):
            iou_mat[i,j] = box_iou(tr.box, db)
    used_tr=set(); used_db=set()
    while True:
        i,j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[i,j] < iou_thr: break
        if i in used_tr or j in used_db:
            iou_mat[i,j] = -1; continue
        matches.append((i,j))
        used_tr.add(i); used_db.add(j)
        iou_mat[i,:] = -1; iou_mat[:,j] = -1
    unmatched_dets = [j for j in range(len(det_boxes)) if j not in used_db]
    unmatched_trks = [i for i in range(len(tracks)) if i not in used_tr]
    return matches, unmatched_dets, unmatched_trks

def extract_roi(gray, box):
    """Вернёт ROI (uint8) по боксу; None, если он пустой/вышел за границы."""
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, gray.shape[1]-1))
    y1 = max(0, min(y1, gray.shape[0]-1))
    x2 = max(0, min(x2, gray.shape[1]-1))
    y2 = max(0, min(y2, gray.shape[0]-1))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    roi = gray[y1:y2, x1:x2]
    return roi


# ---------- оценка capacity ----------

def estimate_capacity(zone_poly, parked_boxes, pack_k=0.68, fallback=8):
    area_zone = abs(cv2.contourArea(zone_poly.astype(np.float32)))
    if len(parked_boxes) == 0:
        return fallback
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in parked_boxes]
    S_car = float(np.median(areas))
    if S_car <= 0 or area_zone <= 0:
        return fallback
    cap = int(np.floor((area_zone * pack_k) / S_car))
    # разумные границы
    return int(max(1, min(1000, cap)))

# ---------- основной пайплайн ----------

def main():
    ap = argparse.ArgumentParser("Parking zones by OpenVINO YOLO")
    ap.add_argument("--model", required=True, type=str, help="OpenVINO IR .xml")
    ap.add_argument("--source", required=True, type=str, help="video/mjpeg/hls url or file")
    ap.add_argument("--zones-json", required=True, type=str, help="JSON с полями zones -> список из 4хточечных полигонов")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="AUTO")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=10.0, help="сколько секунд собирать данные")
    ap.add_argument("--fps", type=float, default=3.0, help="целевая частота обработки")
    ap.add_argument("--dwell", type=float, default=8.0, help="минимум секунд, чтобы считать припаркованным")
    ap.add_argument("--static_thr", type=float, default=3.0, help="макс средняя |разница яркости| (0..255) для статичности")
    ap.add_argument("--grace", type=float, default=4.0, help="секунд держать трек без детекций")
    ap.add_argument("--miss_max", type=int, default=10, help="сколько кадров подряд без матчей живёт трек")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model_xml = Path(args.model).expanduser().resolve()
    if not model_xml.exists():
        print(f"[ERR] model xml not found: {model_xml}", file=sys.stderr); sys.exit(1)

    zones = zones_from_json(Path(args.zones_json))
    if len(zones)==0:
        print("[ERR] zones empty", file=sys.stderr); sys.exit(2)

    # маски зон для быстрого ROI
    zone_masks = []
    frame_probe = None

    # OpenVINO
    ie = Core()
    model = ie.read_model(str(model_xml))
    compiled = ie.compile_model(model, args.device)
    input_tensor = compiled.inputs[0]

    # Video
    cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[ERR] cannot open source: {args.source}", file=sys.stderr); sys.exit(3)

    # подготовим размеры кадра
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERR] cannot read first frame", file=sys.stderr); sys.exit(4)
    H, W = frame.shape[:2]
    frame_probe = frame.copy()
    for poly in zones:
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.reshape(-1,1,2).astype(np.int32)], 255)
        zone_masks.append(mask)

    # Треки по зонам
    zone_tracks = [[] for _ in zones]  # список списков Track
    parked_ids_by_zone = [set() for _ in zones]

    t_end = time.time() + args.duration
    next_tick = 0.0
    last_gray = cv2.cvtColor(frame_probe, cv2.COLOR_BGR2GRAY)

    last_frame_to_show = frame_probe.copy()
    total_processed = 0

    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok or frame is None:
            # попытка переподключиться
            cap.release(); time.sleep(0.5)
            cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
            if not cap.isOpened(): break
            continue

        now = time.time()
        if next_tick > now:
            # пропускаем кадры до следующего такта
            continue
        next_tick = now + 1.0/max(0.1, args.fps)

        H, W = frame.shape[:2]
        img, r, (dw,dh) = letterbox(frame, args.imgsz)
        img_rgb = img[:, :, ::-1].astype(np.float32)/255.0
        if input_tensor.element_type.type_name == "f16":
            img_rgb = img_rgb.astype(np.float16)
        else:
            img_rgb = img_rgb.astype(np.float32)
        img_nchw = np.transpose(img_rgb, (2,0,1))[None, ...]

        # infer
        req = compiled.create_infer_request()
        req.set_tensor(compiled.inputs[0], Tensor(img_nchw))
        req.infer()
        outputs = [req.get_tensor(o) for o in compiled.outputs]

        parsed = parse_with_embedded_nms(outputs, conf_thres=args.conf)
        if parsed is None:
            boxes, scores, cls_ids = parse_raw_yolo_outputs(outputs, args.conf, args.iou, car_only=True, car_id=0)
        else:
            boxes, scores, cls_ids = parsed
            m = (cls_ids.astype(int)==0)
            boxes, scores = boxes[m], scores[m]

        # rescale boxes back
        if boxes.shape[0] > 0:
            boxes[:,[0,2]] -= dw; boxes[:,[1,3]] -= dh
            boxes[:,:4] /= r
            boxes[:,0::2] = boxes[:,0::2].clip(0, W-1)
            boxes[:,1::2] = boxes[:,1::2].clip(0, H-1)

        # фильтрация по зонам и трекинг
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # подготовим рисование
        canvas = frame.copy()
        draw_zones(canvas, zones)

        # распределим детекции по зонам по центру бокса
        dets_by_zone = [[] for _ in zones]
        scores_by_zone = [[] for _ in zones]
        for b,s in zip(boxes, scores):
            cx = 0.5*(b[0]+b[2]); cy = 0.5*(b[1]+b[3])
            for zi, poly in enumerate(zones):
                if point_in_poly(cx, cy, poly):
                    dets_by_zone[zi].append(b.copy())
                    scores_by_zone[zi].append(float(s))
                    break

        # обновим треки по зонам
        for zi, (poly, mask) in enumerate(zip(zones, zone_masks)):
            tracks = zone_tracks[zi]
            dets = dets_by_zone[zi]
            scs = scores_by_zone[zi]

            # матчинг
            matches, um_det, um_tr = match_tracks(tracks, dets, iou_thr=0.3)

            # обновить сматченные
            # обновить сматченные
            for ti, dj in matches:
                tr = tracks[ti]
                tr.box = dets[dj].astype(np.float32)
                tr.score = max(tr.score, scs[dj])
                tr.last_ts = now
                tr.hits += 1
                tr.miss = 0

                # безопасный расчёт "статичности"
                roi_new = extract_roi(gray, tr.box)
                if roi_new is not None and tr.prev_roi_gray is not None:
                    prev = tr.prev_roi_gray
                    # приводим прошлый ROI к размеру текущего
                    if prev.shape != roi_new.shape:
                        prev = cv2.resize(prev, (roi_new.shape[1], roi_new.shape[0]), interpolation=cv2.INTER_AREA)
                    # средняя |разница| яркости
                    d = float(cv2.absdiff(roi_new, prev).mean())
                    tr.motion_buf.append(d)
                tr.prev_roi_gray = roi_new

            # создать новые треки
            for dj in um_det:
                tr = Track(dets[dj], now, scs[dj])
                tr.prev_roi_gray = extract_roi(gray, tr.box)
                tracks.append(tr)

            # увеличить miss у непромаченных, удалить умершие
            new_tracks=[]
            for idx,tr in enumerate(tracks):
                if idx in [t for t,_ in matches]:
                    new_tracks.append(tr); continue
                tr.miss += 1
                if tr.miss <= args.miss_max and tr.seen_recently(now, args.grace):
                    new_tracks.append(tr)
            zone_tracks[zi] = new_tracks

            # определим припаркованные
            parked = []
            for tr in zone_tracks[zi]:
                age_ok = tr.age(now) >= args.dwell
                motion_ok = (len(tr.motion_buf)>0 and np.mean(tr.motion_buf) <= args.static_thr)
                if age_ok and motion_ok:
                    parked.append(tr)

            parked_ids_by_zone[zi] = set([tr.id for tr in parked])

            # рисование
            # все текущие детекции:
            if len(dets)>0:
                draw_boxes(canvas, np.array(dets), np.array(scs), color=(0,255,0), label="car")
            # треки припаркованных
            for tr in parked:
                x1,y1,x2,y2 = [int(round(v)) for v in tr.box]
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,128,255), 2)
                txt = f"id{tr.id} parked {int(tr.age(now))}s, motion={np.mean(tr.motion_buf):.1f}"
                cv2.putText(canvas, txt, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(canvas, txt, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # подписи зоны: capacity/occupied/free
            parked_boxes = [tr.box for tr in zone_tracks[zi] if tr.id in parked_ids_by_zone[zi]]
            capacity = estimate_capacity(poly, parked_boxes, pack_k=0.68, fallback=8)
            occ = len(parked_ids_by_zone[zi])
            free = max(0, capacity - occ)
            # центр для подписи
            cX = int(np.mean(poly[:,0])); cY = int(np.mean(poly[:,1]))
            txt = f"Z{zi+1}: occ={occ} free={free} cap={capacity}"
            cv2.putText(canvas, txt, (cX+6, cY+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(canvas, txt, (cX+6, cY+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        last_frame_to_show = canvas
        total_processed += 1

    cap.release()

    # финальная статистика
    results = []
    for zi, poly in enumerate(zones):
        parked = [tr for tr in zone_tracks[zi] if tr.id in parked_ids_by_zone[zi]]
        boxes = [tr.box for tr in parked]
        capz = estimate_capacity(poly, boxes, pack_k=0.68, fallback=8)
        occ = len(parked)
        free = max(0, capz - occ)
        results.append({
            "zone": zi+1,
            "occupied": int(occ),
            "free": int(free),
            "capacity_est": int(capz)
        })

    # выводим статистику в консоль
    print(json.dumps({"zones": results, "processed_frames": total_processed}, ensure_ascii=False, indent=2))

    # показываем последний кадр
    if args.show and last_frame_to_show is not None:
        cv2.imshow("Parking analysis (press any key)", last_frame_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
