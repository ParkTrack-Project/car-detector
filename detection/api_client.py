import sys
import requests


def fetch_next_camera(
    http_session: requests.Session,
    base_api_url: str,
    timeout_seconds: float = 5.0
):
    """
    Получает данные о следующей камере через API:
        GET {BASE_API_URL}/cameras/next
    """
    request_url = base_api_url.rstrip("/") + "/cameras/next"
    response = http_session.get(request_url, timeout=timeout_seconds)
    response.raise_for_status()
    camera_data = response.json()
    return camera_data


def fetch_zones_for_camera(
    http_session: requests.Session,
    base_api_url: str,
    camera_id: int,
    timeout_seconds: float = 5.0
):
    """
    Получает список зон для заданной камеры через API:
        GET {BASE_API_URL}/zones?camera_id=<camera_id>
    """
    request_url = base_api_url.rstrip("/") + "/zones"
    response = http_session.get(
        request_url,
        params={"camera_id": camera_id},
        timeout=timeout_seconds
    )
    response.raise_for_status()
    zones_data = response.json()

    if not isinstance(zones_data, list):
        raise RuntimeError("Ожидался список зон от /zones?camera_id=...")

    return zones_data


def update_zone_occupancy(
    http_session: requests.Session,
    base_api_url: str,
    zone_id: int,
    occupied_count: int,
    zone_confidence: float,
    timeout_seconds: float = 5.0
):
    """
    Обновляет информацию о занятости конкретной зоны через API:
        PUT {BASE_API_URL}/zones/<zone_id>
    """
    request_url = base_api_url.rstrip("/") + f"/zones/{zone_id}"
    request_payload = {
        "occupied": int(occupied_count),
        "confidence": float(zone_confidence),
    }

    response = http_session.put(
        request_url,
        json=request_payload,
        timeout=timeout_seconds
    )

    if not (200 <= response.status_code < 300):
        print(
            f"[WARN] zone {zone_id} update failed: "
            f"{response.status_code} {response.text}",
            file=sys.stderr
        )
