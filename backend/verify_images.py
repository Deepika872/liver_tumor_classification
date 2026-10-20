
from pathlib import Path
import cv2

checks = [
    ("Original", Path("dataset/original"), "volume-9_61.png"),
    ("Preprocessed", Path("preprocessed_dataset"), "volume-9_61.png"),
    ("Liver mask", Path("dataset/liver_masks"), "segmentation-9_livermask_61.png"),
    ("Lesion mask", Path("dataset/lesion_masks"), "segmentation-9_lesionmask_61.png"),
]

for label, folder, filename in checks:
    path = folder / filename
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"{label}: FILE NOT FOUND OR UNREADABLE - {path}")
    else:
        print(f"{label}: OK | Shape: {image.shape} | File: {filename}")
