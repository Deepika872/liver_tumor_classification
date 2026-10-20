
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "dataset" / "original"
OUTPUT_DIR = BASE_DIR / "preprocessed_dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def reduce_noise(image):
    # Bilateral filter: reduce noise while preserving edges
    return cv2.bilateralFilter(
        image,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

def main():
    if not INPUT_DIR.exists():
        print(f"Dataset folder not found: {INPUT_DIR.resolve()}")
        return

    processed_count = 0

    for image_path in INPUT_DIR.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        filtered_image = reduce_noise(image)

        # Preserve subfolders and filenames
        relative_path = image_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if cv2.imwrite(str(output_path), filtered_image):
            processed_count += 1

    print(f"Noise reduction completed: {processed_count} images")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
