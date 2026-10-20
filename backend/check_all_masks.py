
from pathlib import Path
import cv2

root = Path(r"C:\Users\billu\Downloads\archive\dataset_6\dataset_6")

for mask_type in ["livermask", "lesionmask"]:
    files = list(root.glob(f"segmentation-*_{mask_type}_*.png"))

    nonempty = 0
    empty = 0
    unreadable = 0

    print(f"\nChecking {mask_type}...")
    print("Files found:", len(files))

    for file in files:
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

        if image is None:
            unreadable += 1
        elif cv2.countNonZero(image) == 0:
            empty += 1
        else:
            nonempty += 1

    print("Total files:", len(files))
    print("Non-empty:", nonempty)
    print("Completely black:", empty)
    print("Unreadable:", unreadable)

print("\nScan completed!")
