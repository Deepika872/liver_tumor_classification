
from pathlib import Path
import cv2
import re

folder = Path(r"C:\Users\billu\Downloads\archive\dataset_6\dataset_6")

originals = sorted(folder.glob("volume-*.png"))
valid = []
missing = []

for original in originals:
    match = re.fullmatch(r"volume-(\d+)_(\d+)\.png", original.name)

    if not match:
        continue

    study, slice_no = match.groups()

    liver_path = folder / f"segmentation-{study}_livermask_{slice_no}.png"
    lesion_path = folder / f"segmentation-{study}_lesionmask_{slice_no}.png"

    if not liver_path.exists() or not lesion_path.exists():
        missing.append(original.name)
        continue

    liver = cv2.imread(str(liver_path), cv2.IMREAD_GRAYSCALE)
    lesion = cv2.imread(str(lesion_path), cv2.IMREAD_GRAYSCALE)

    if liver is None or lesion is None:
        missing.append(original.name)
        continue

    liver_pixels = cv2.countNonZero(liver)
    lesion_pixels = cv2.countNonZero(lesion)

    if liver_pixels > 0:
        valid.append(original.name)
        print(
            f"{original.name} | Liver pixels: {liver_pixels} "
            f"| Lesion pixels: {lesion_pixels}"
        )

print("\nTotal original images:", len(originals))
print("Images with non-empty liver masks:", len(valid))
print("Missing/unreadable mask pairs:", len(missing))
