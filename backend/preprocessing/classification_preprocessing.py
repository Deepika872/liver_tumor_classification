
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "dataset" / "classification_dataset"
OUTPUT_DIR = BASE_DIR / "dataset" / "classification_preprocessed"

CLASSES = ["benign", "malignant", "normal"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def reduce_noise(image):
    # Bilateral filter reduces noise while preserving edges
    return cv2.bilateralFilter(
        image,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )


def main():
    if not INPUT_DIR.exists():
        print(f"Dataset folder not found: {INPUT_DIR}")
        return

    total_processed = 0

    for class_name in CLASSES:
        class_input = INPUT_DIR / class_name
        class_output = OUTPUT_DIR / class_name

        if not class_input.exists():
            print(f"Missing class folder: {class_name}")
            continue

        class_output.mkdir(parents=True, exist_ok=True)
        class_count = 0

        for image_path in class_input.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not read: {image_path}")
                continue

            filtered = reduce_noise(image)

            relative_path = image_path.relative_to(class_input)
            output_path = class_output / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if cv2.imwrite(str(output_path), filtered):
                class_count += 1
                total_processed += 1
            else:
                print(f"Could not save: {output_path}")

        print(f"{class_name}: {class_count} images processed")

    print(f"\nNoise reduction completed: {total_processed} images")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
