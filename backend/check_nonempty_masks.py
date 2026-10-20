
from pathlib import Path
import cv2

folder = Path(r"C:\Users\billu\Downloads\archive\dataset_6\dataset_6")

print("Sample non-empty liver masks:")

count = 0

for file in sorted(folder.glob("segmentation-*livermask*.png")):
    image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

    if image is not None:
        white_pixels = cv2.countNonZero(image)

        if white_pixels > 0:
            print(file.name, "| White pixels:", white_pixels)
            count += 1

            if count == 10:
                break

if count == 0:
    print("No non-empty liver masks found!")
