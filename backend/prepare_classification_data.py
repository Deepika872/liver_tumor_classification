
from pathlib import Path
from collections import defaultdict
import hashlib
import shutil

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_DIR = BASE_DIR / "dataset" / "classification_dataset"
PROCESSED_DIR = BASE_DIR / "dataset" / "classification_preprocessed"
OUTPUT_DIR = BASE_DIR / "dataset" / "classification_splits"

CLASSES = ["benign", "malignant", "normal"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
RANDOM_SEED = 42


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not ORIGINAL_DIR.exists() or not PROCESSED_DIR.exists():
        print("Original or preprocessed dataset folder is missing.")
        return

    # Build a list of unique original images per class.
    records = []
    duplicate_count = 0

    for class_name in CLASSES:
        original_class_dir = ORIGINAL_DIR / class_name
        processed_class_dir = PROCESSED_DIR / class_name

        if not original_class_dir.exists() or not processed_class_dir.exists():
            print(f"Missing folder for class: {class_name}")
            return

        seen_hashes = set()

        original_images = sorted(
            path for path in original_class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for original_path in original_images:
            digest = file_hash(original_path)

            if digest in seen_hashes:
                duplicate_count += 1
                print(f"Skipping exact duplicate: {original_path.name}")
                continue

            seen_hashes.add(digest)

            relative_path = original_path.relative_to(original_class_dir)
            processed_path = processed_class_dir / relative_path

            if not processed_path.exists():
                print(f"Processed image missing: {processed_path}")
                return

            records.append({
                "class_name": class_name,
                "original": original_path,
                "processed": processed_path,
            })

    counts = {
        class_name: sum(r["class_name"] == class_name for r in records)
        for class_name in CLASSES
    }

    print(f"\nExact duplicates skipped: {duplicate_count}")
    print("Unique images by class:", counts)

    # Make stratified train / validation / test splits.
    labels = [record["class_name"] for record in records]

    try:
        train_records, temp_records = train_test_split(
            records,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=labels,
        )

        temp_labels = [record["class_name"] for record in temp_records]

        val_records, test_records = train_test_split(
            temp_records,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=temp_labels,
        )
    except ValueError as error:
        print("Could not create stratified splits:", error)
        print("No split folders were created.")
        return

    # Clear only the generated split folder, never the source datasets.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    splits = {
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    }

    for split_name, split_records in splits.items():
        for record in split_records:
            destination_dir = (
                OUTPUT_DIR / split_name / record["class_name"]
            )
            destination_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                record["processed"],
                destination_dir / record["processed"].name,
            )

    print("\nDataset split completed.")

    for split_name, split_records in splits.items():
        print(f"\n{split_name.upper()}: {len(split_records)} images")

        for class_name in CLASSES:
            count = sum(
                record["class_name"] == class_name
                for record in split_records
            )
            print(f"  {class_name}: {count}")

    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
