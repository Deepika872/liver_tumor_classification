"""
Smart Liver Tumor Classification Using EfficientNetV2B0

Classes:
    0 - Benign
    1 - Malignant
    2 - Normal

Academic research prototype only.
Not intended for clinical diagnosis.
"""

import os
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "classification_splits"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "efficientnetv2_liver_tumor.keras"

CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

RESULTS_PATH = MODEL_DIR / "evaluation_results.json"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

INITIAL_EPOCHS = 10

FINE_TUNE_EPOCHS = 5

SEED = 42

CLASS_NAMES = [
    "benign",
    "malignant",
    "normal",
]

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Reduce unnecessary TensorFlow log messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ==========================================================
# 2. CHECK DATASET FOLDERS
# ==========================================================

def check_dataset_folders():
    """Check that the expected dataset folders exist."""

    required_folders = [
        DATASET_DIR / "train",
        DATASET_DIR / "validation",
        DATASET_DIR / "test",
    ]

    for folder in required_folders:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        for class_name in CLASS_NAMES:
            class_folder = folder / class_name

            if not class_folder.is_dir():
                raise FileNotFoundError(
                    f"Class folder not found: {class_folder}"
                )

    print("\nDataset folder verification completed.")

    for split in ["train", "validation", "test"]:
        print(f"\n{split.upper()} dataset:")

        for class_name in CLASS_NAMES:
            class_folder = DATASET_DIR / split / class_name

            image_count = sum(
                1
                for path in class_folder.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".bmp"}
            )

            print(f"  {class_name}: {image_count} images")

# ==========================================================
# 3. LOAD DATASET
# ==========================================================

def load_datasets():
    """Load training, validation and test images."""

    train_dir = DATASET_DIR / "train"
    validation_dir = DATASET_DIR / "validation"
    test_dir = DATASET_DIR / "test"

    print("\nLoading training dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    print("\nLoading validation dataset...")

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        str(validation_dir),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nLoading test dataset...")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nClass order:", train_ds.class_names)

    # EfficientNetV2B0 includes its own preprocessing layer
    # by default. Keep image pixel values in the 0-255 range.
    # Do not divide images by 255 here.

    # Prefetch helps improve the input pipeline.
    train_ds = train_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    validation_ds = validation_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return train_ds, validation_ds, test_ds

# ==========================================================
# 4. BUILD EFFICIENTNETV2B0 MODEL
# ==========================================================

def build_model():
    """Build a transfer-learning classification model."""

    print("\nLoading EfficientNetV2B0 pretrained weights...")

    base_model = (
        tf.keras.applications.EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            input_shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3,
            ),
            include_preprocessing=True,
        )
    )

    # Initially freeze the pretrained feature extractor.
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        name="input_image",
    )

    # Data augmentation is applied only during training.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="data_augmentation",
    )

    x = augmentation(inputs)

    # Keep Batch Normalization layers in inference mode.
    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="classification_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetV2B0_Liver_Classification",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel architecture created successfully.")

    model.summary()

    return model, base_model

# ==========================================================
# 5. CREATE CALLBACKS
# ==========================================================

def create_callbacks():
    """Create callbacks to control training."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    return callbacks

# ==========================================================
# 6. TRAIN CLASSIFICATION HEAD
# ==========================================================

def train_initial_model(
    model,
    train_ds,
    validation_ds,
    callbacks,
):
    """Train the classification layers first."""

    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING CLASSIFICATION HEAD")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    return history

# ==========================================================
# 7. OPTIONAL FINE-TUNING
# ==========================================================

def fine_tune_model(
    model,
    base_model,
    train_ds,
    validation_ds,
):
    """Fine-tune a small number of pretrained layers."""

    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING")
    print("=" * 60)

    # Unfreeze the pretrained model.
    base_model.trainable = True

    # Freeze most layers and fine-tune only the final layers.
    fine_tune_at = max(
        0,
        len(base_model.layers) - 20,
    )

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Keep Batch Normalization layers frozen for stability.
    for layer in base_model.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    return history

# ==========================================================
# 8. EVALUATE ON TEST DATA
# ==========================================================

def evaluate_model(model, test_ds):
    """Evaluate the model using the separate test dataset."""

    print("\n" + "=" * 60)
    print("FINAL TEST DATA EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1,
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test accuracy percentage: {test_accuracy * 100:.2f}%")

    true_labels = []
    predicted_labels = []

    for images, labels in test_ds:
        probabilities = model.predict(
            images,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        true_labels.extend(
            labels.numpy().tolist()
        )

        predicted_labels.extend(
            predictions.tolist()
        )

    print("\nClassification Report:")

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
    )

    print("Confusion Matrix:")
    print(matrix)

    results = {
        "model": "EfficientNetV2B0",
        "class_names": CLASS_NAMES,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(
            test_accuracy * 100
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "test_sample_count": len(true_labels),
        "medical_validation": (
            "Not clinically validated. Research prototype only."
        ),
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(f"\nEvaluation results saved to:\n{RESULTS_PATH}")

# ==========================================================
# 9. SAVE CLASS NAMES
# ==========================================================

def save_class_names():
    """Save class order for later prediction API use."""

    with open(
        CLASS_NAMES_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            CLASS_NAMES,
            file,
            indent=4,
        )

    print(f"\nClass names saved to:\n{CLASS_NAMES_PATH}")

# ==========================================================
# 10. MAIN TRAINING PIPELINE
# ==========================================================

def main():
    print("=" * 60)
    print("SMART LIVER TUMOR CLASSIFICATION")
    print("EfficientNetV2B0 Training Pipeline")
    print("=" * 60)

    print("\nTensorFlow version:", tf.__version__)

    print(
        "Available GPUs:",
        tf.config.list_physical_devices("GPU"),
    )

    check_dataset_folders()

    train_ds, validation_ds, test_ds = load_datasets()

    model, base_model = build_model()

    callbacks = create_callbacks()

    # Phase 1: train classification layers.
    train_initial_model(
        model,
        train_ds,
        validation_ds,
        callbacks,
    )

    # Phase 2: fine-tune the final pretrained layers.
    fine_tune_model(
        model,
        base_model,
        train_ds,
        validation_ds,
    )

    # Reload the best saved checkpoint.
    if MODEL_PATH.exists():
        print("\nLoading the best saved model...")

        model = tf.keras.models.load_model(
            str(MODEL_PATH)
        )

    # Evaluate only after training and fine-tuning.
    evaluate_model(
        model,
        test_ds,
    )

    save_class_names()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 60)

    print(f"\nTrained model:\n{MODEL_PATH}")

    print(
        "\nReminder: Results are preliminary and "
        "must not be used for medical diagnosis."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted by user. "
            "Check whether a model checkpoint was saved."
        )

    except Exception as error:
        print("\nTRAINING FAILED")
        print("Error:", error)

        raise