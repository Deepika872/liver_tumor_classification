
from pathlib import Path
import re
import cv2

original_dir = Path("dataset/original")
liver_dir = Path("dataset/liver_masks")
lesion_dir = Path("dataset/lesion_masks")

errors = []

for image_path in sorted(original_dir.glob("*.png")):
    match = re.fullmatch(r"volume-(\d+)_(\d+)\.png", image_path.name)

    if not match:
        errors.append(f"Unexpected filename: {image_path.name}")
        continue

    study, slice_no = match.groups()

    liver_path = liver_dir / f"segmentation-{study}_livermask_{slice_no}.png"
    lesion_path = lesion_dir / f"segmentation-{study}_lesionmask_{slice_no}.png"

    if not liver_path.exists():
        errors.append(f"Missing liver mask: {liver_path.name}")
        continue

    if not lesion_path.exists():
        errors.append(f"Missing lesion mask: {lesion_path.name}")
        continue

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    liver = cv2.imread(str(liver_path), cv2.IMREAD_GRAYSCALE)
    lesion = cv2.imread(str(lesion_path), cv2.IMREAD_GRAYSCALE)

    if image is None or liver is None or lesion is None:
        errors.append(f"Could not read files for {image_path.name}")
    elif image.shape != liver.shape or image.shape != lesion.shape:
        errors.append(f"Dimension mismatch: {image_path.name}")
        
print(f"Original images checked: {len(list(original_dir.glob('*.png')))}")
print(f"Issues found: {len(errors)}")

for error in errors:
    print(error)

if not errors:
    print("All image-mask filenames and dimensions match!")
