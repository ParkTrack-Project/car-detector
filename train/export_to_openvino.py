from ultralytics import YOLO
from pathlib import Path

# путь к твоему чекпоинту
pt_path = Path("./runs/cars_ft_s0/weights/best.pt")

# где сохранить (опционально)
out_dir = pt_path.parent

# загрузка и экспорт
model = YOLO(str(pt_path))
export_path = model.export(
    format="openvino",   # OpenVINO IR
    imgsz=640,           # или 512/736/… кратно 32
    half=True,           # FP16 (рекомендуется для CPU/GPU)
    dynamic=False,       # статический размер входа (быстрее)
    opset=12,            # стабильный opset
    verbose=True
)

print(f"Exported to: {export_path}")  # обычно …/best_openvino_model
