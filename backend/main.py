from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from services.prediction_service import predict_image


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="Smart Liver Tumor Classification API",
    description=(
        "EfficientNetV2B0-based liver image classification "
        "research prototype"
    ),
    version="1.1.0",
)


# =====================================================
# CORS CONFIGURATION
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# CONFIGURATION
# =====================================================

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_FORMATS = {"JPEG", "PNG"}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}


# =====================================================
# HOME ENDPOINT
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Liver Tumor Classification API is running",
        "model": "EfficientNetV2B0",
        "status": "connected",
    }


# =====================================================
# HEALTH CHECK ENDPOINT
# =====================================================

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "model": "EfficientNetV2B0",
        "classes": [
            "benign",
            "malignant",
            "normal",
        ],
    }


# =====================================================
# IMAGE PREDICTION ENDPOINT
# =====================================================

@app.post("/api/v1/predictions")
async def receive_image(
    image: UploadFile = File(...),
):
    # -------------------------------------------------
    # 1. Validate content type
    # -------------------------------------------------

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image.",
        )

    # -------------------------------------------------
    # 2. Read uploaded file safely
    # -------------------------------------------------

    contents = await image.read(
        MAX_FILE_SIZE + 1
    )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB.",
        )

    # -------------------------------------------------
    # 3. Validate actual image format
    # -------------------------------------------------

    try:
        with Image.open(
            BytesIO(contents)
        ) as uploaded_image:

            if uploaded_image.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG and PNG images are supported.",
                )

            uploaded_image.verify()

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except OSError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is corrupted or invalid.",
        )

    # -------------------------------------------------
    # 4. Generate prediction
    # -------------------------------------------------

    try:
        result = predict_image(contents)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        print(
            "Prediction error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend terminal.",
        )

    # -------------------------------------------------
    # 5. Return result to frontend
    # -------------------------------------------------

    return {
        "message": "Prediction completed successfully",
        "filename": image.filename,
        "status": "success",
        "model": "EfficientNetV2B0",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "class_probabilities": result["class_probabilities"],
        "disclaimer": result["disclaimer"],
    }