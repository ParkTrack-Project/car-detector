#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage:

python detect_mjpeg_openvino.py \
  --model /opt/yoloapp/models/best_openvino_model/model.xml \
  --source http://109.236.111.203/mjpg/video.mjpg \
  --imgsz 640 \
  --conf 0.15 \
  --iou 0.5 \
  --car_only \
  --show
"""

import argparse
import sys
from pathlib import Path
import time
import yaml
import cv2
import numpy as np
from openvino import Core, Tensor

# ------------------------
# Utils
# ------------------------

def load_class_names(model_dir: Path):
    """
    Пытаемся прочитать имена классов из metadata.yaml (Ultralytics export).
    Если не вышло — по умолчанию единственный класс 'car'.
    """
    meta = model_dir / "metadata.yaml"
    if meta.exists():
        try:
            data = yaml.safe_load(meta.read_text())
            names = data.get("names") or data.get("classes") or None
            if isinstance(names, dict):
                # {"0": "car", ...} -> сортируем по ключу
                names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
            if isinstance(names, list) and len(names) > 0:
                return names
        except Exception:
            pass
    return ["car"]

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """
    Изменение размера с сохранением пропорций и паддингом под квадрат new_shape x new_shape.
    Возвращает: изображение, масштаб (r), паддинги (dw, dh).
    """
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh_to_xyxy(x):
    # x:[N,4] as [cx, cy, w, h] -> [x1,y1,x2,y2]
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_thres=0.5):
    """
    Простая NMS. boxes: [N,4] (xyxy), scores: [N]
    Возвращает индексы оставшихся боксов.
    """
    if len(boxes) == 0:
        return []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def draw_boxes(frame, boxes, scores, cls_ids, names, color=(0, 255, 0)):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, cls_ids):
        label = names[int(c)] if 0 <= int(c) < len(names) else str(int(c))
        pct = int(round(float(s) * 100))
        txt = f"{label} {pct}%"
        x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1i, y1i - th - 6), (x1i + tw + 4, y1i), color, -1)
        cv2.putText(frame, txt, (x1i + 2, y1i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

# ------------------------
# Postprocess variants
# ------------------------

def parse_with_embedded_nms(output_tensors, conf_thres):
    """
    Попытка распарсить модель, где NMS уже внутри графа (часто у Ultralytics OpenVINO).
    Ищем выход [1, N, 6] со значениями [x1,y1,x2,y2,score,class_id].
    """
    for t in output_tensors:
        arr = t.data if hasattr(t, "data") else np.asarray(t)
        a = np.array(arr)
        if a.ndim == 3 and a.shape[-1] == 6:
            det = a[0]  # [N, 6]
            if det.size == 0:
                return np.empty((0,4)), np.empty((0,)), np.empty((0,))
            boxes = det[:, :4]
            scores = det[:, 4]
            cls_ids = det[:, 5]
            m = scores >= conf_thres
            return boxes[m], scores[m], cls_ids[m]
    return None  # не нашли подходящий формат

def parse_raw_yolo_outputs(output_tensors, conf_thres, iou_thres, car_only=False, car_id=0):
    """
    Обработка «сырых» выходов YOLOv8/11/12: [1, 4+nc, N] либо [1, N, 4+nc].
    Берём лучший класс, применяем порог и NMS.
    """
    # Выбираем наибольший по числу элементов выход как предсказания
    arrays = []
    for t in output_tensors:
        arr = t.data if hasattr(t, "data") else np.asarray(t)
        arrays.append(np.array(arr))
    pred = max(arrays, key=lambda a: a.size)

    if pred.ndim == 3:
        if pred.shape[1] in (84,85,4+80,4+1,4+10) and pred.shape[0] == 1:
            # [1, 4+nc, N] -> (N, 4+nc)
            pred = np.transpose(pred[0], (1, 0))
        elif pred.shape[2] in (84,85) and pred.shape[0] == 1:
            # [1, N, 4+nc] -> (N, 4+nc)
            pred = pred[0]
        elif pred.shape[0] == 1 and pred.shape[1] == 1:
            pred = pred[0,0]
    elif pred.ndim == 2:
        pass
    else:
        raise RuntimeError("Неизвестный формат выхода модели.")

    if pred.shape[1] < 5:
        raise RuntimeError("Выход модели не похож на YOLO-предсказания (меньше 5 каналов).")

    # Разделяем боксы и классы
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]  # [N, nc] (в YOLOv8 это уже class confidences)

    # лучший класс
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    # фильтр порогом
    m = scores >= conf_thres
    boxes_xywh = boxes_xywh[m]
    scores = scores[m]
    cls_ids = cls_ids[m]

    # при необходимости — оставить только машины (class_id == 0 по твоей модели)
    if car_only:
        car_mask = (cls_ids == car_id)
        boxes_xywh = boxes_xywh[car_mask]
        scores = scores[car_mask]
        cls_ids = cls_ids[car_mask]

    # xywh -> xyxy
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    # NMS
    keep = nms(boxes_xyxy, scores, iou_thres=iou_thres)
    if len(keep) == 0:
        return np.empty((0,4)), np.empty((0,)), np.empty((0,))
    keep = np.array(keep, dtype=int)
    return boxes_xyxy[keep], scores[keep], cls_ids[keep]

# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenVINO YOLO detector for MJPEG stream")
    parser.add_argument("--model", type=str, required=True,
                        help="Путь к model.xml (OpenVINO IR)")
    parser.add_argument("--source", type=str, default="http://109.236.111.203/mjpg/video.mjpg",
                        help="MJPEG URL или путь к видео")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер входа модели (кратно 32)")
    parser.add_argument("--device", type=str, default="AUTO", help="OPENVINO target: AUTO/GPU/CPU")
    parser.add_argument("--conf", type=float, default=0.15, help="порог уверенности (0..1)")
    parser.add_argument("--iou", type=float, default=0.5, help="порог IoU для NMS")
    parser.add_argument("--show", action="store_true", help="Показывать окно (imshow). По умолчанию включено если есть GUI.")
    parser.add_argument("--save", type=str, default="", help="Путь для сохранения вывода (mp4). Пусто — не сохранять.")
    parser.add_argument("--car_only", action="store_true", help="Фильтровать только класс 'car' (id=0). Полезно если в модели много классов.")
    args = parser.parse_args()

    model_xml = Path(args.model).expanduser().resolve()
    if not model_xml.exists():
        print(f"[ERR] model.xml не найден: {model_xml}", file=sys.stderr)
        sys.exit(1)

    model_dir = model_xml.parent
    names = load_class_names(model_dir)
    # По твоей постановке обычно один класс 'car'
    car_id = 0

    # OpenVINO load
    ie = Core()
    model = ie.read_model(model=model_xml)
    compiled = ie.compile_model(model=model, device_name=args.device)
    input_tensor = compiled.inputs[0]
    in_shape = list(input_tensor.shape)
    h_in, w_in = int(in_shape[2]), int(in_shape[3])

    # Видеоввод
    cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[ERR] Не удалось открыть источник: {args.source}", file=sys.stderr)
        sys.exit(2)

    # Видеовывод (опционально)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Определим размер по первому кадру
        ret, probe = cap.read()
        if not ret:
            print("[ERR] Не удалось прочитать кадр для определения размера.", file=sys.stderr)
            sys.exit(3)
        h0, w0 = probe.shape[:2]
        writer = cv2.VideoWriter(args.save, fourcc, 25.0, (w0, h0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # вернуться в начало потока, для MJPEG не критично

    # main loop
    win_name = "Detections (q to quit)"
    can_show = args.show or (not bool(cv2.getBuildInformation().find("with GTK+") == -1))
    fps_avg = 0.0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # на mjpg-потоках иногда помогает повторное открытие
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("[WARN] Попытка переподключения к камере не удалась.")
                break
            continue

        h0, w0 = frame.shape[:2]

        # препроцессинг
        img, r, (dw, dh) = letterbox(frame, new_shape=args.imgsz)
        img = img[:, :, ::-1]  # BGR->RGB
        img = img.astype(np.float32) / 255.0

        # привести к типу входа модели (f16 или f32)
        if input_tensor.element_type.type_name == "f16":
            img = img.astype(np.float16)
        else:
            img = img.astype(np.float32)

        # NCHW
        img = np.transpose(img, (2, 0, 1))[None, ...]

        # инференс
        infer_req = compiled.create_infer_request()
        infer_req.set_tensor(compiled.inputs[0], Tensor(img))  # передаём ov.Tensor
        infer_req.infer()
        outputs = [infer_req.get_tensor(o) for o in compiled.outputs]

        # постпроцессинг — сначала пробуем «встроенный NMS»
        parsed = parse_with_embedded_nms(outputs, conf_thres=args.conf)
        if parsed is None:
            # сырые выходы YOLO
            boxes, scores, cls_ids = parse_raw_yolo_outputs(
                outputs, conf_thres=args.conf, iou_thres=args.iou,
                car_only=args.car_only or (len(names) == 1 and names[0].lower() == "car"),
                car_id=car_id
            )
        else:
            boxes, scores, cls_ids = parsed
            # если надо оставить только автомобили
            if args.car_only or (len(names) == 1 and names[0].lower() == "car"):
                m = (cls_ids.astype(int) == car_id)
                boxes, scores, cls_ids = boxes[m], scores[m], cls_ids[m]

        # масштабирование координат обратно в размер исходного кадра
        if boxes.shape[0] > 0:
            # координаты были посчитаны относительно letterbox-картины (imgsz x imgsz)
            # снимаем паддинги и масштаб
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes[:, :4] /= r
            # клип
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0 - 1)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0 - 1)

        # рисуем
        draw_boxes(frame, boxes, scores, cls_ids, names, color=(0, 255, 0))

        # FPS
        fps = 1.0 / max(1e-6, (time.time() - t0))
        fps_avg = fps_avg * 0.9 + fps * 0.1 if fps_avg > 0 else fps
        t0 = time.time()
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        if can_show or args.show:
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
