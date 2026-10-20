
from pathlib import Path
import hashlib
import cv2

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "classification_dataset"
CLASSES = ["benign", "malignant", "normal"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

def main():
    print("CLASSIFICATION DATASET VERIFICATION")
    print("-" * 40)

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        return

    total_images = 0
    unreadable_images = []
    dimension_counts = {}
    hashes = {}

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"{class_name}: FOLDER MISSING")
            continue

        images = [
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\n{class_name}: {len(images)} images")
        total_images += len(images)

        for image_path in images:
            image = cv2.imread(str(image_path))

            if image is None:
                unreadable_images.append(str(image_path))
                continue

            height, width = image.shape[:2]
            size = f"{width}x{height}"
            dimension_counts[size] = dimension_counts.get(size, 0) + 1

            digest = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()

            hashes.setdefault(digest, []).append(
                f"{class_name}/{image_path.name}"
            )

    duplicates = [
        paths for paths in hashes.values()
        if len(paths) > 1
    ]

    print("\n" + "-" * 40)
    print(f"Total images: {total_images}")
    print(f"Readable images: {total_images - len(unreadable_images)}")
    print(f"Unreadable images: {len(unreadable_images)}")
    print(f"Image dimensions: {dimension_counts}")
    print(f"Duplicate image groups: {len(duplicates)}")

    if unreadable_images:
        print("\nUnreadable files:")
        for path in unreadable_images:
            print(path)

    if duplicates:
        print("\nDuplicate groups:")
        for group in duplicates:
            print(group)

    print("\nVerification completed.")

if __name__ == "__main__":
    main()
