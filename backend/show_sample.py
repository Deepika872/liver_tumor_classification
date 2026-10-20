
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

files = [
    ("Original CT", "dataset/original/volume-9_61.png"),
    ("Preprocessed CT", "preprocessed_dataset/volume-9_61.png"),
    ("Liver Mask", "dataset/liver_masks/segmentation-9_livermask_61.png"),
    ("Lesion Mask", "dataset/lesion_masks/segmentation-9_lesionmask_61.png"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for ax, (title, filename) in zip(axes, files):
    path = Path(filename)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"FILE NOT FOUND: {path.resolve()}")
        ax.set_title(f"{title}\nFile missing")
        ax.axis("off")
        continue

    print(
        f"{title}: min={image.min()}, max={image.max()}, "
        f"unique_values={np.unique(image)[:20]}"
    )

    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("dataset_sample_check.png", dpi=150)
plt.show()
