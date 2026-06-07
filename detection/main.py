import sys
from pathlib import Path
import json
import traceback
import cv2

from cli import parse_arguments
from pipeline import run_single_frame_pipeline


def main():
    """
    Точка входа: парсит аргументы, запускает пайплайн, выводит JSON
    и, при необходимости, сохраняет/показывает кадр.
    """
    args = parse_arguments()

    try:
        result_payload, visualization_frame_bgr, source_frame = run_single_frame_pipeline(args)
    except Exception as exception:
        print(f"[ERR] pipeline failed: {type(exception).__name__}: {exception}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Печатаем JSON в stdout
    print(json.dumps(result_payload, ensure_ascii=False))

    # Сохранение изображения (если указано)
    if args.out_img:
        # Берём camera_id из результата пайплайна
        camera_id = result_payload.get("camera_id")

        out_dir = Path(args.out_img).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if camera_id is None:
            # Фолбэк, если вдруг camera_id нет (на всякий случай)
            annotated_output_image_path = out_dir / "annotated.jpg"
            source_image_path = out_dir / "source.jpg"
        else:
            annotated_output_image_path = out_dir / f"{camera_id}.jpg"
            source_image_path = out_dir / f"{camera_id}_source.jpg"

        annotated_success_write = cv2.imwrite(str(annotated_output_image_path), visualization_frame_bgr)
        if not annotated_success_write:
            print(
                f"[WARN] cannot save annotated image to {annotated_output_image_path}",
                file=sys.stderr
            )
        source_success_write = cv2.imwrite(str(source_image_path), source_frame)
        if not source_success_write:
            print(
                f"[WARN] cannot save source image to {annotated_output_image_path}",
                file=sys.stderr
            )

    # Показ окна (если нужно)
    if args.show:
        cv2.imshow(
            "Curved zones on distorted image (single frame, API)",
            visualization_frame_bgr
        )
        cv2.waitKey(1500)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
