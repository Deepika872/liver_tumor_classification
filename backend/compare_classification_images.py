
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_DIR = BASE_DIR / "dataset" / "classification_dataset"
PROCESSED_DIR = BASE_DIR / "dataset" / "classification_preprocessed"

CLASSES = ["benign", "malignant", "normal"]

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for row, class_name in enumerate(CLASSES):
    folder = ORIGINAL_DIR / class_name

    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not images:
        print(f"No images found: {class_name}")
        continue

    original_path = images[0]
    processed_path = PROCESSED_DIR / class_name / original_path.relative_to(folder)

    original = cv2.imread(str(original_path))
    processed = cv2.imread(str(processed_path))

    if original is None or processed is None:
        print(f"Could not read image: {class_name}")
        continue

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    axes[row, 0].imshow(original)
    axes[row, 0].set_title(f"{class_name.title()} - Original")
    axes[row, 1].imshow(processed)
    axes[row, 1].set_title(f"{class_name.title()} - Noise Reduced")

    axes[row, 0].axis("off")
    axes[row, 1].axis("off")

plt.tight_layout()
plt.show()
