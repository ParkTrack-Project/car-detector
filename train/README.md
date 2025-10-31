# Экспорт `best.pt` в OpenVINO

Пример:
```shell
yolo export model=./train/runs/train_yolov125/weights/best.pt format=openvino imgsz=640 half=True dynamic=False opset=12
```