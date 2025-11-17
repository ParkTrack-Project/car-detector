import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
from openvino import Core, Tensor
import requests

from yolo_utils import (
    load_class_names,
    letterbox,
    parse_with_embedded_nms,
    parse_raw_yolo_outputs,
)
from calibration import (
    load_calibration_from_dict,
    compute_fullview_rectified_camera_matrix,
)
from geometry import (
    build_curved_polygon_from_anchors,
    signed_depth_to_polygon,
    overlap_ratio_box_in_polygon,
)
from visualization import (
    vivid_palette,
    draw_polygon_outline,
    put_text_outline,
    draw_legend_bottom_left,
    draw_box_with_alpha,
)
from api_client import (
    fetch_next_camera,
    fetch_zones_for_camera,
    update_zone_occupancy,
)


# ---------- Вспомогательные шаги пайплайна ----------

def setup_http_session(api_token: str) -> requests.Session:
    """
    Создаёт HTTP-сессию с нужным заголовком Authorization.

    Аргументы:
        api_token (str): Bearer токен для доступа к API.

    Возвращает:
        requests.Session: Готовая сессия.
    """
    session = requests.Session()
    if api_token:
        session.headers.update({"Authorization": f"Bearer {api_token}"})
    return session


def fetch_camera_and_calibration(
    http_session: requests.Session,
    base_api_url: str,
):
    """
    Запрашивает следующую камеру и парсит калибровку.

    Аргументы:
        http_session (requests.Session): HTTP-сессия.
        base_api_url (str): Базовый URL API.

    Возвращает:
        tuple:
            camera_id (int),
            video_source_url (str),
            frame_width_from_calib (int),
            frame_height_from_calib (int),
            camera_matrix (np.ndarray),
            distortion_coefficients (np.ndarray),
            rectified_camera_matrix (np.ndarray | None)
    """
    camera_info = fetch_next_camera(http_session, base_api_url)

    camera_id = int(camera_info["camera_id"])
    video_source_url = camera_info["source"]
    calibration_raw = camera_info["calib"]

    (
        calibration_image_width,
        calibration_image_height,
        camera_matrix,
        distortion_coefficients,
        rectified_camera_matrix_opt,
        _,
    ) = load_calibration_from_dict(calibration_raw)

    return (
        camera_id,
        video_source_url,
        calibration_image_width,
        calibration_image_height,
        camera_matrix,
        distortion_coefficients,
        rectified_camera_matrix_opt,
    )


def grab_first_frame(
    video_source_url: str,
    timeout_seconds: float = 5.0
) -> np.ndarray:
    """
    Считывает первый доступный кадр из видеопотока.

    Аргументы:
        video_source_url (str): URL/путь к видеоисточнику.
        timeout_seconds (float): Таймаут на ожидание кадра.

    Возвращает:
        np.ndarray: Первый кадр в формате BGR.

    Исключения:
        RuntimeError: если кадр получить не удалось.
    """
    video_capture = cv2.VideoCapture(video_source_url, cv2.CAP_FFMPEG)
    if not video_capture.isOpened():
        raise RuntimeError(f"cannot open source: {video_source_url}")

    first_frame_bgr = None
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        read_success, frame_bgr = video_capture.read()
        if read_success and frame_bgr is not None:
            first_frame_bgr = frame_bgr
            break
        time.sleep(0.05)

    video_capture.release()

    if first_frame_bgr is None:
        raise RuntimeError("failed to read frame from source")

    return first_frame_bgr


def adjust_camera_matrix_to_frame_size(
    camera_matrix: np.ndarray,
    calibration_image_width: int,
    calibration_image_height: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Подгоняет матрицу камеры под фактический размер кадра, если он
    отличается от размера, использованного при калибровке.

    Аргументы:
        camera_matrix (np.ndarray): Матрица камеры K (3x3).
        calibration_image_width (int): Ширина, использованная при калибровке.
        calibration_image_height (int): Высота, использованная при калибровке.
        frame_width (int): Реальная ширина кадра.
        frame_height (int): Реальная высота кадра.

    Возвращает:
        np.ndarray: Обновлённая матрица камеры K (3x3).
    """
    if (
        calibration_image_width == frame_width
        and calibration_image_height == frame_height
    ):
        return camera_matrix

    scale_x = frame_width / calibration_image_width
    scale_y = frame_height / calibration_image_height

    camera_matrix = camera_matrix.copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 2] *= scale_y

    return camera_matrix


def build_curved_zones_from_api(
    http_session: requests.Session,
    base_api_url: str,
    camera_id: int,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    rectified_camera_matrix: np.ndarray,
    samples_per_edge: int,
):
    """
    Получает зоны через API и строит для каждой изогнутый полигон.

    Аргументы:
        http_session (requests.Session): HTTP-сессия.
        base_api_url (str): Базовый URL API.
        camera_id (int): Идентификатор камеры.
        camera_matrix (np.ndarray): Матрица камеры K.
        distortion_coefficients (np.ndarray): Коэффициенты дисторсии D.
        rectified_camera_matrix (np.ndarray): Матрица newK.
        samples_per_edge (int): Плотность дискретизации границ зоны.

    Возвращает:
        tuple:
            curved_zone_polygons (list[np.ndarray]): список полигонов (каждый (M, 2)).
            zone_identifiers (list[int]): соответствующие zone_id.
    """
    zones_from_api = fetch_zones_for_camera(http_session, base_api_url, camera_id)

    if not zones_from_api:
        print(f"[WARN] no zones for camera {camera_id}", file=sys.stderr)

    curved_zone_polygons = []
    zone_identifiers = []

    for zone_description in zones_from_api:
        zone_points = zone_description.get("points") or []
        if len(zone_points) < 3:
            # игнорируем некорректные зоны из API
            continue

        distorted_anchors_pixels = np.array(
            [[point["x"], point["y"]] for point in zone_points],
            dtype=np.float64
        )

        curved_zone_polygon = build_curved_polygon_from_anchors(
            distorted_anchors_pixels,
            camera_matrix,
            distortion_coefficients,
            rectified_camera_matrix,
            samples_per_edge=max(8, samples_per_edge),
        )

        curved_zone_polygons.append(curved_zone_polygon)
        zone_identifiers.append(int(zone_description["zone_id"]))

    if not curved_zone_polygons:
        print("[WARN] no valid polygons built from zones", file=sys.stderr)

    return curved_zone_polygons, zone_identifiers


def run_openvino_inference_on_frame(
    frame_bgr: np.ndarray,
    model_xml_path: Path,
    device: str,
    img_size: int,
    confidence_threshold: float,
    car_only: bool,
):
    """
    Запускает инференс YOLO-модели OpenVINO на одном кадре.

    Аргументы:
        frame_bgr (np.ndarray): Входной кадр BGR.
        model_xml_path (Path): Путь к model.xml.
        device (str): Устройство OpenVINO (AUTO, CPU, GPU, ...).
        img_size (int): Размер входного изображения модели.
        confidence_threshold (float): Порог уверенности.
        car_only (bool): Оставлять только класс 'car' (если он есть).

    Возвращает:
        tuple:
            bounding_boxes_xyxy (np.ndarray),
            detection_scores (np.ndarray),
            detection_class_ids (np.ndarray),
            class_names (list[str]),
            resize_ratio (float),
            padding_width (float),
            padding_height (float)
    """
    if not model_xml_path.exists():
        raise RuntimeError(f"model.xml not found: {model_xml_path}")

    class_names = load_class_names(model_xml_path.parent)
    car_class_id = 0  # по умолчанию считаем, что класс 'car' = 0

    openvino_core = Core()
    openvino_model = openvino_core.read_model(model=str(model_xml_path))
    compiled_model = openvino_core.compile_model(
        model=openvino_model,
        device_name=device
    )
    model_input_tensor = compiled_model.inputs[0]

    resized_frame_bgr, resize_ratio, (padding_width, padding_height) = letterbox(
        frame_bgr,
        target_shape=img_size
    )

    resized_frame_rgb = resized_frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    if model_input_tensor.element_type.type_name == "f16":
        resized_frame_rgb = resized_frame_rgb.astype(np.float16)
    else:
        resized_frame_rgb = resized_frame_rgb.astype(np.float32)

    model_input_blob = np.transpose(resized_frame_rgb, (2, 0, 1))[None, ...]

    infer_request = compiled_model.create_infer_request()
    infer_request.set_tensor(compiled_model.inputs[0], Tensor(model_input_blob))
    infer_request.infer()

    model_outputs = [infer_request.get_tensor(output_node) for output_node in compiled_model.outputs]

    parsed_outputs = parse_with_embedded_nms(
        model_outputs,
        confidence_threshold=confidence_threshold
    )

    if parsed_outputs is None:
        bounding_boxes_xyxy, detection_scores, detection_class_ids = parse_raw_yolo_outputs(
            model_outputs,
            confidence_threshold=confidence_threshold,
            car_only=car_only or (len(class_names) == 1 and class_names[0].lower() == "car"),
            car_class_id=car_class_id,
            nms_iou_threshold=0.5
        )
    else:
        bounding_boxes_xyxy, detection_scores, detection_class_ids = parsed_outputs
        if car_only or (len(class_names) == 1 and class_names[0].lower() == "car"):
            car_mask = (detection_class_ids.astype(int) == car_class_id)
            bounding_boxes_xyxy = bounding_boxes_xyxy[car_mask]
            detection_scores = detection_scores[car_mask]
            detection_class_ids = detection_class_ids[car_mask]

    return (
        bounding_boxes_xyxy,
        detection_scores,
        detection_class_ids,
        class_names,
        resize_ratio,
        padding_width,
        padding_height,
    )


def restore_boxes_to_original_frame(
    bounding_boxes_xyxy: np.ndarray,
    resize_ratio: float,
    padding_width: float,
    padding_height: float,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Переводит координаты боксов после letterbox обратно в систему
    координат исходного кадра.

    Аргументы:
        bounding_boxes_xyxy (np.ndarray): Боксы в координатах letterbox.
        resize_ratio (float): Масштаб.
        padding_width (float): Горизонтальный паддинг.
        padding_height (float): Вертикальный паддинг.
        frame_width (int): Ширина исходного кадра.
        frame_height (int): Высота исходного кадра.

    Возвращает:
        np.ndarray: Боксы в координатах исходного кадра.
    """
    if bounding_boxes_xyxy.shape[0] == 0:
        return bounding_boxes_xyxy

    boxes = bounding_boxes_xyxy.copy()
    boxes[:, [0, 2]] -= padding_width
    boxes[:, [1, 3]] -= padding_height
    boxes[:, :4] /= resize_ratio

    boxes[:, 0::2] = boxes[:, 0::2].clip(0, frame_width - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, frame_height - 1)

    return boxes


def assign_detections_to_zones(
    bounding_boxes_xyxy: np.ndarray,
    detection_scores: np.ndarray,
    detection_class_ids: np.ndarray,
    curved_zone_polygons: List[np.ndarray],
    zone_identifiers: List[int],
    frame_width: int,
    frame_height: int,
):
    """
    Присваивает каждую детекцию (машину) конкретной зоне
    на основе:
        - центр бокса должен быть внутри полигона;
        - выбираем зону с максимальным overlap.

    Аргументы:
        bounding_boxes_xyxy, detection_scores, detection_class_ids: результаты детекции.
        curved_zone_polygons (list[np.ndarray]): список полигонов зон.
        zone_identifiers (list[int]): идентификаторы зон.
        frame_width (int): ширина кадра.
        frame_height (int): высота кадра.

    Возвращает:
        tuple:
            zone_statistics (list[dict]): статистика по зонам (occupied, cars, confidence=0.0).
            car_assigned_zone_indices (list[int]): индекс зоны для каждой детекции или -1.
    """
    zone_statistics: List[Dict[str, Any]] = []
    for index in range(len(curved_zone_polygons)):
        zone_statistics.append({
            "id": int(zone_identifiers[index]),
            "occupied": 0,
            "cars": [],
            "confidence": 0.0,
        })

    car_assigned_zone_indices = [-1] * bounding_boxes_xyxy.shape[0]

    for detection_index, (bounding_box_xyxy, detection_score, detection_class_id) in enumerate(
        zip(bounding_boxes_xyxy, detection_scores, detection_class_ids)
    ):
        box_center_x = float((bounding_box_xyxy[0] + bounding_box_xyxy[2]) * 0.5)
        box_center_y = float((bounding_box_xyxy[1] + bounding_box_xyxy[3]) * 0.5)
        box_center = (box_center_x, box_center_y)

        best_zone_index = -1
        best_overlap_ratio = 0.0

        for zone_index, curved_polygon in enumerate(curved_zone_polygons):
            depth_inside_polygon = signed_depth_to_polygon(
                box_center,
                curved_polygon
            )
            if depth_inside_polygon <= 0.0:
                continue

            overlap_ratio = overlap_ratio_box_in_polygon(
                bounding_box_xyxy,
                curved_polygon,
                frame_height,
                frame_width
            )
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_zone_index = zone_index

        assigned_zone_index = best_zone_index if best_overlap_ratio > 0.0 else -1
        car_assigned_zone_indices[detection_index] = assigned_zone_index

        if assigned_zone_index >= 0:
            depth_inside_assigned_polygon = signed_depth_to_polygon(
                box_center,
                curved_zone_polygons[assigned_zone_index]
            )
            overlap_ratio_assigned = overlap_ratio_box_in_polygon(
                bounding_box_xyxy,
                curved_zone_polygons[assigned_zone_index],
                frame_height,
                frame_width
            )

            zone_statistics[assigned_zone_index]["occupied"] += 1
            zone_statistics[assigned_zone_index]["cars"].append({
                "det_index": detection_index,
                "center": [box_center_x, box_center_y],
                "box": [
                    float(bounding_box_xyxy[0]),
                    float(bounding_box_xyxy[1]),
                    float(bounding_box_xyxy[2]),
                    float(bounding_box_xyxy[3]),
                ],
                "score": float(detection_score),
                "class_id": int(detection_class_id),
                "depth_px": float(depth_inside_assigned_polygon),
                "overlap_ratio": float(overlap_ratio_assigned),
            })

    return zone_statistics, car_assigned_zone_indices


def compute_zone_confidences(zone_statistics: List[Dict[str, Any]]) -> None:
    """
    Заполняет поле "confidence" для каждой зоны: усреднённое score * overlap_ratio
    по всем машинам в зоне.

    Аргументы:
        zone_statistics (list[dict]): Статистика по зонам.
    """
    for zone_info in zone_statistics:
        cars_in_zone = zone_info["cars"]
        if not cars_in_zone:
            zone_info["confidence"] = 0.0
            continue

        weighted_scores_sum = 0.0
        for car_info in cars_in_zone:
            weighted_scores_sum += car_info["score"] * car_info["overlap_ratio"]
        zone_info["confidence"] = float(weighted_scores_sum / len(cars_in_zone))


def render_visualization_frame(
    base_frame_bgr: np.ndarray,
    curved_zone_polygons: List[np.ndarray],
    zone_statistics: List[Dict[str, Any]],
    car_assigned_zone_indices: List[int],
    bounding_boxes_xyxy: np.ndarray,
    detection_scores: np.ndarray,
    detection_class_ids: np.ndarray,
    class_names: List[str],
    zone_colors_bgr: List[Tuple[int, int, int]],
    zone_outline_thickness: int,
    car_alpha: float,
) -> np.ndarray:
    """
    Строит финальный кадр с визуализацией зон и машин.

    Возвращает:
        np.ndarray: Кадр BGR с разметкой.
    """
    visualization_frame_bgr = base_frame_bgr.copy()

    # Зоны
    for zone_index, curved_polygon in enumerate(curved_zone_polygons):
        zone_color_bgr = zone_colors_bgr[zone_index]
        draw_polygon_outline(
            visualization_frame_bgr,
            curved_polygon,
            color_bgr=zone_color_bgr,
            thickness=zone_outline_thickness
        )

        polygon_center_xy = np.mean(curved_polygon, axis=0)
        zone_label_text = f"Zone {zone_statistics[zone_index]['id']}: {zone_statistics[zone_index]['occupied']}"
        text_x = int(round(polygon_center_xy[0]))
        text_y = int(round(polygon_center_xy[1]))
        put_text_outline(
            visualization_frame_bgr,
            zone_label_text,
            (max(0, text_x - 40), max(12, text_y)),
            zone_color_bgr
        )

    # Машины
    for detection_index, (bounding_box_xyxy, detection_score, detection_class_id) in enumerate(
        zip(bounding_boxes_xyxy, detection_scores, detection_class_ids)
    ):
        if 0 <= int(detection_class_id) < len(class_names):
            detection_class_name = class_names[int(detection_class_id)]
        else:
            detection_class_name = str(int(detection_class_id))

        detection_score_percent = int(round(float(detection_score) * 100))
        detection_label_text = f"{detection_class_name} {detection_score_percent}%"

        assigned_zone_index = car_assigned_zone_indices[detection_index]

        if assigned_zone_index >= 0:
            box_edge_color_bgr = zone_colors_bgr[assigned_zone_index]
            box_fill_color_bgr = zone_colors_bgr[assigned_zone_index]
            draw_box_with_alpha(
                visualization_frame_bgr,
                bounding_box_xyxy,
                detection_label_text,
                edge_color_bgr=box_edge_color_bgr,
                fill_color_bgr=box_fill_color_bgr,
                alpha=car_alpha,
                thickness=2
            )
        else:
            draw_box_with_alpha(
                visualization_frame_bgr,
                bounding_box_xyxy,
                detection_label_text,
                edge_color_bgr=(0, 255, 0),
                fill_color_bgr=None,
                alpha=0.0,
                thickness=2
            )

        box_center_x = int(round((bounding_box_xyxy[0] + bounding_box_xyxy[2]) * 0.5))
        box_center_y = int(round((bounding_box_xyxy[1] + bounding_box_xyxy[3]) * 0.5))
        cv2.circle(visualization_frame_bgr, (box_center_x, box_center_y), 3, (0, 0, 0), -1)
        cv2.circle(visualization_frame_bgr, (box_center_x, box_center_y), 2, (255, 255, 255), -1)

    legend_entries = [
        f"Zone {zone_info['id']}: {zone_info['occupied']}"
        for zone_info in zone_statistics
    ]
    draw_legend_bottom_left(
        visualization_frame_bgr,
        legend_entries,
        zone_colors_bgr,
        margin_pixels=10,
        padding_pixels=8,
        line_height_pixels=22,
        background_alpha=0.5
    )

    return visualization_frame_bgr


def build_result_payload(
    camera_id: int,
    video_source_url: str,
    frame_width: int,
    frame_height: int,
    samples_per_edge: int,
    base_api_url: str,
    zone_statistics: List[Dict[str, Any]],
    bounding_boxes_xyxy: np.ndarray,
) -> Dict[str, Any]:
    """
    Собирает JSON-пейлоад результата, аналогичный тому, что раньше печатался в main.

    Возвращает:
        dict: Готовый JSON-словарь результата.
    """
    result_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "source": str(video_source_url),
        "zones": zone_statistics,
        "totals": {
            "cars_detected": int(bounding_boxes_xyxy.shape[0]),
            "cars_in_zones": int(sum(zone_info["occupied"] for zone_info in zone_statistics)),
        },
        "meta": {
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "samples_per_edge": int(max(8, samples_per_edge)),
            "base_api_url": base_api_url,
        },
    }
    return result_payload


def push_zone_updates_to_api(
    http_session: requests.Session,
    base_api_url: str,
    zone_statistics: List[Dict[str, Any]],
) -> None:
    """
    Проходит по всем зонам и отправляет их занятость и confidence в API.

    Аргументы:
        http_session (requests.Session): HTTP-сессия.
        base_api_url (str): Базовый URL API.
        zone_statistics (list[dict]): Статистика по зонам.
    """
    for zone_info in zone_statistics:
        try:
            update_zone_occupancy(
                http_session,
                base_api_url,
                zone_id=zone_info["id"],
                occupied_count=zone_info["occupied"],
                zone_confidence=zone_info["confidence"],
            )
        except Exception as exception:
            print(
                f"[WARN] failed to update zone {zone_info['id']}: {exception}",
                file=sys.stderr
            )


# ---------- Высокоуровневый единый шаг ----------

def run_single_frame_pipeline(args):
    """
    Запускает полный пайплайн "один кадр" от API до результата.

    Аргументы:
        args (argparse.Namespace): Аргументы командной строки.

    Возвращает:
        tuple:
            result_payload (dict): JSON-словарь результата.
            visualization_frame_bgr (np.ndarray): Визуализированный кадр.
    """
    base_api_url = args.base_api_url

    # 1. HTTP-сессия
    http_session = setup_http_session(args.api_token)

    # 2. Камера + калибровка
    (
        camera_id,
        video_source_url,
        calibration_image_width,
        calibration_image_height,
        camera_matrix,
        distortion_coefficients,
        rectified_camera_matrix_opt,
    ) = fetch_camera_and_calibration(http_session, base_api_url)

    # 3. Первый кадр
    first_frame_bgr = grab_first_frame(video_source_url)
    frame_height, frame_width = first_frame_bgr.shape[:2]

    # 4. Подгонка матрицы камеры
    camera_matrix = adjust_camera_matrix_to_frame_size(
        camera_matrix,
        calibration_image_width,
        calibration_image_height,
        frame_width,
        frame_height,
    )

    # 5. rectified_camera_matrix
    if rectified_camera_matrix_opt is not None:
        rectified_camera_matrix = rectified_camera_matrix_opt
    else:
        rectified_camera_matrix = compute_fullview_rectified_camera_matrix(
            frame_width,
            frame_height,
            camera_matrix,
            distortion_coefficients
        )

    # 6. Зоны + изогнутые полигоны
    curved_zone_polygons, zone_identifiers = build_curved_zones_from_api(
        http_session,
        base_api_url,
        camera_id,
        camera_matrix,
        distortion_coefficients,
        rectified_camera_matrix,
        samples_per_edge=args.samples_per_edge,
    )
    zone_colors_bgr = vivid_palette(len(curved_zone_polygons))

    # 7. Инференс
    model_xml_path = Path(args.model).expanduser().resolve()
    (
        bounding_boxes_xyxy,
        detection_scores,
        detection_class_ids,
        class_names,
        resize_ratio,
        padding_width,
        padding_height,
    ) = run_openvino_inference_on_frame(
        first_frame_bgr,
        model_xml_path=model_xml_path,
        device=args.device,
        img_size=args.imgsz,
        confidence_threshold=args.conf,
        car_only=args.car_only,
    )

    # 8. Перенос боксов в координаты оригинального кадра
    bounding_boxes_xyxy = restore_boxes_to_original_frame(
        bounding_boxes_xyxy,
        resize_ratio=resize_ratio,
        padding_width=padding_width,
        padding_height=padding_height,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    # 9. Назначение машин зонам
    zone_statistics, car_assigned_zone_indices = assign_detections_to_zones(
        bounding_boxes_xyxy,
        detection_scores,
        detection_class_ids,
        curved_zone_polygons,
        zone_identifiers,
        frame_width,
        frame_height,
    )

    # 10. confidence по зонам
    compute_zone_confidences(zone_statistics)

    # 11. Визуализация
    visualization_frame_bgr = render_visualization_frame(
        first_frame_bgr,
        curved_zone_polygons,
        zone_statistics,
        car_assigned_zone_indices,
        bounding_boxes_xyxy,
        detection_scores,
        detection_class_ids,
        class_names,
        zone_colors_bgr,
        zone_outline_thickness=args.zone_thickness,
        car_alpha=args.car_alpha,
    )

    # 12. JSON-результат
    result_payload = build_result_payload(
        camera_id,
        video_source_url,
        frame_width,
        frame_height,
        samples_per_edge=args.samples_per_edge,
        base_api_url=base_api_url,
        zone_statistics=zone_statistics,
        bounding_boxes_xyxy=bounding_boxes_xyxy,
    )

    # 13. Отправка в API
    push_zone_updates_to_api(http_session, base_api_url, zone_statistics)

    return result_payload, visualization_frame_bgr
