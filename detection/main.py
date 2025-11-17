import sys
from pathlib import Path
import json

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
        result_payload, visualization_frame_bgr = run_single_frame_pipeline(args)
    except Exception as exception:
        # Центральная точка обработки ошибок
        print(f"[ERR] pipeline failed: {exception}", file=sys.stderr)
        sys.exit(1)

    # Печатаем JSON в stdout
    print(json.dumps(result_payload, ensure_ascii=False))

    # Сохранение изображения (если указано)
    if args.out_img:
        output_image_path = Path(args.out_img).expanduser().resolve()
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        success_write = cv2.imwrite(str(output_image_path), visualization_frame_bgr)
        if not success_write:
            print(
                f"[WARN] cannot save image to {output_image_path}",
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
