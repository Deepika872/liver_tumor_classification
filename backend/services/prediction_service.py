import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "efficientnetv2_liver_tumor.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR
    / "models"
    / "class_names.json"
)

IMAGE_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

model = tf.keras.models.load_model(
    str(MODEL_PATH)
)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8",
) as file:
    CLASS_NAMES = json.load(file)

if len(CLASS_NAMES) != model.output_shape[-1]:
    raise ValueError(
        "Number of class names does not match model output."
    )

print("EfficientNetV2B0 model loaded successfully.")
print("Classes:", CLASS_NAMES)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image and prepare it for the model.

    EfficientNetV2B0 includes its own preprocessing layer,
    so pixel values remain in the 0-255 range.
    """

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    try:
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )

        image = tf.image.resize(
            image,
            IMAGE_SIZE,
        )

        image = tf.cast(
            image,
            tf.float32,
        )

        image = tf.expand_dims(
            image,
            axis=0,
        )

        return image.numpy()

    except Exception as exc:
        raise ValueError(
            "Unable to read the uploaded image."
        ) from exc


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_image(image_bytes: bytes) -> dict:
    """
    Generate a model prediction for an uploaded image.

    This is an academic research prototype and is not
    validated for clinical diagnosis.
    """

    image = preprocess_image(image_bytes)

    probabilities = model.predict(
        image,
        verbose=0,
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    class_probabilities = {
        class_name: round(
            float(probabilities[index]),
            6,
        )
        for index, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 6),
        "class_probabilities": class_probabilities,
        "disclaimer": (
            "Research prototype only. "
            "Not intended for clinical diagnosis."
        ),
    }