"""
python train_yolo.py
  --data "./merged_dataset/data.yaml"
  --model-size s
  --epochs 100
  --imgsz 640
  --batch 8
  --device mps
  --workers 4
  --project runs
  --name cars_ft_s

"""

import argparse
import sys
import zipfile
from pathlib import Path
import yaml
from glob import glob

def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def ensure_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
        return
    except Exception as e:
        print("[INFO] Пакет ultralytics не найден или устарел. Пытаюсь установить/обновить...", flush=True)
        import subprocess, sys as _sys
        # Свежая версия нужна для YOLO12
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
        from ultralytics import YOLO  # noqa: F401

def unzip_if_needed(data_path: Path, out_dir: Path) -> Path:
    if data_path.is_file() and data_path.suffix.lower() == ".zip":
        target = out_dir / data_path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(data_path), "r") as zf:
            zf.extractall(target)
        return target
    return data_path

def find_data_yaml(root: Path) -> Path:
    # Часто лежит в корне, но на всякий случай ищем рекурсивно
    candidates = list(root.rglob("data.yaml"))
    if not candidates:
        die("Не найден файл data.yaml внутри датасета.")
    # Берем самый верхний по уровню вложенности
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]

def count_images(img_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if not img_dir.exists():
        return 0
    return sum(1 for p in img_dir.rglob("*") if p.suffix.lower() in exts)

def stats_from_yaml(data_yaml: Path):
    with open(data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    base = data_yaml.parent
    def to_path(v):
        # Пути в YAML могут быть относительными или абсолютными
        p = Path(v)
        return p if p.is_absolute() else (base / p).resolve()

    train_img = to_path(y.get("train", ""))  # обычно .../train/images
    val_img   = to_path(y.get("val", y.get("valid", "")))  # на случай "valid"
    test_img  = to_path(y.get("test", ""))

    # Если в YAML указан путь до images напрямую — считаем там
    # Если указан путь до папки сплита — добавим /images
    def normalize_images_dir(p: Path) -> Path:
        if p.name.lower() == "images":
            return p
        # если внутри есть images — используем его
        cand = p / "images"
        return cand if cand.exists() else p

    train_img = normalize_images_dir(train_img)
    val_img   = normalize_images_dir(val_img)
    test_img  = normalize_images_dir(test_img) if test_img else test_img

    train_n = count_images(train_img)
    val_n   = count_images(val_img) if val_img else 0
    test_n  = count_images(test_img) if test_img else 0

    total = train_n + val_n + test_n
    return {
        "train": (train_img, train_n),
        "val":   (val_img,   val_n),
        "test":  (test_img,  test_n),
        "total": total,
        "names": y.get("names"),
        "nc":    y.get("nc")
    }

def print_stats(title: str, s: dict):
    print(f"\n=== Датасет: {title} ===")
    print(f"Классы (nc): {s['nc']}, names: {s['names']}")
    def line(name):
        p, n = s[name]
        if p:
            print(f"{name:>5}: {n:5d}  ({p})")
        else:
            print(f"{name:>5}: {0:5d}  (нет)")
    line("train")
    line("val")
    line("test")
    print(f"ИТОГО: {s['total']} изображений")

def main():
    parser = argparse.ArgumentParser(description="Обучение YOLOv12 на вашем датасете (YOLO формат).")
    parser.add_argument("--data", required=True,
                        help="Путь к архиву .zip, к папке датасета или напрямую к data.yaml")
    parser.add_argument("--model-size", default="n", choices=list("nsm lx".replace(" ", "")),
                        help="Размер модели: n|s|m|l|x (по умолчанию n)")
    parser.add_argument("--model", type=str, default=None,
                        help="Путь к .pt/.yaml или ID модели (например yolo12s.pt). "
                             "Если указан, игнорирует --model-size/--from-scratch.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="напр. 0, 0,1, 'cpu' (по умолчанию авто)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="train_yolov12")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Старт с конфигурации .yaml вместо предобученных .pt")
    parser.add_argument("--resume", action="store_true", help="Возобновить последнее обучение (если есть)")
    parser.add_argument("--patience", type=int, default=50, help="Ранний стоп")
    parser.add_argument("--save-pred", action="store_true", help="Сохранить пример предсказаний после обучения")
    args = parser.parse_args()

    data_arg = Path(args.data).expanduser().resolve()
    if not data_arg.exists():
        die(f"Файл/папка не существует: {data_arg}")

    # Подготовка данных
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if data_arg.suffix.lower() == ".yaml":
        data_yaml = data_arg
        root = data_yaml.parent
    else:
        root = unzip_if_needed(data_arg, datasets_dir)
        data_yaml = find_data_yaml(root)

    # Печать статистики по датасету
    stats = stats_from_yaml(data_yaml)
    print_stats(root.name, stats)

    # Импорт ultralytics (и установка при необходимости)
    ensure_ultralytics()
    from ultralytics import YOLO

    size = args.model_size.lower().strip()
    if size not in list("nsm lx".replace(" ", "")):
        die("model-size должен быть одним из: n,s,m,l,x")

    try:
        if args.resume:
            ckpt = Path(args.project) / args.name / "weights" / "last.pt"
            print(f"\n[INFO] Возобновляю обучение из последнего чекпоинта...\n")
            model = YOLO(str(ckpt))
            model.train(resume=True)
            sys.exit(0)
        if args.model:
            model = YOLO(args.model)  # тут может быть runs/.../last.pt или yolo12s.pt
        else:
            model_id = f"yolo12{size}.pt" if not args.from_scratch else f"yolo12{size}.yaml"
            print(f"\n[INFO] Загружаю модель: {model_id}")
            model = YOLO(model_id)

            print(f"[INFO] Старт обучения на {args.epochs} эпох, imgsz={args.imgsz}, batch={args.batch}")
            results = model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=args.device or None,
                project=args.project,
                name=args.name,
                patience=args.patience,
                cache="disk",
                deterministic=False,
                # single_cls НЕ ставим — при nc=1 это не требуется
            )

            print("\n[INFO] Валидация на val...")
            model.val(data=str(data_yaml), device=args.device or None)
            if stats["test"][0]:
                print("[INFO] Оценка на test...")
                model.val(data=str(data_yaml), split="test", device=args.device or None)

        print(f"[INFO] Старт обучения на {args.epochs} эпох, imgsz={args.imgsz}, batch={args.batch}")
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device or None,  # mps/ cpu
            project=args.project,
            name=args.name,
            patience=args.patience,
            cache="disk",  # на MPS экономим память
            deterministic=False,  # чтобы не было "Using 0 dataloader workers"
            plots=False
        )

        print("\n[INFO] Валидация на val...")
        model.val(data=str(data_yaml), device=args.device or None)
        if stats["test"][0]:
            print("[INFO] Оценка на test...")
            model.val(data=str(data_yaml), split="test", device=args.device or None)

        # Если есть test — прогоняем и его
        if stats["test"][0]:
            print("[INFO] Оценка на test...")
            model.val(data=str(data_yaml), split="test")

        # Сохранить пример предсказаний
        if args.save_pred:
            # Возьмем несколько изображений из val (или train, если val пуст)
            src_dir = stats["val"][0] or stats["train"][0]
            if src_dir and count_images(src_dir) > 0:
                some_images = []
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    some_images.extend(glob(str(src_dir / ext)))
                    if len(some_images) >= 8:
                        break
                some_images = some_images[:8] if some_images else None
                if some_images:
                    print("[INFO] Сохраняю пример предсказаний...")
                    model.predict(
                        source=some_images,
                        conf=0.25,
                        save=True,
                        project=args.project,
                        name=f"{args.name}_pred"
                    )
            else:
                print("[WARN] Нет изображений для примера предсказаний.")

        print("\nГотово. Чекпоинты и логи см. в папке runs/.")

    except Exception as e:
        die(f"Не удалось запустить YOLOv12. Убедитесь, что установлен свежий ultralytics. Исходная ошибка: {e}")

if __name__ == "__main__":
    main()
