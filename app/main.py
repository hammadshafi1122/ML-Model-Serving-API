
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas import PredictRequest, PredictResponse


# Logging configuration
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("model-serving")

# Resolve the model path relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "artifacts" / "iris_model.joblib"),
    )
)

MODEL: Any = None
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# Iris dataset class labels
LABEL_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
    "0": "setosa",
    "1": "versicolor",
    "2": "virginica",
    "setosa": "setosa",
    "versicolor": "versicolor",
    "virginica": "virginica",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}. "
                "Run python train.py first."
            )

        MODEL = joblib.load(MODEL_PATH)

        if not hasattr(MODEL, "predict"):
            raise ValueError("Loaded model does not support predict().")

        if not hasattr(MODEL, "predict_proba"):
            raise ValueError(
                "Loaded model does not support predict_proba()."
            )

        logger.info("Model loaded successfully from %s", MODEL_PATH)
        yield

    except Exception:
        logger.exception("Failed to load model")
        raise

    finally:
        MODEL = None
        logger.info("Model service shutting down")


app = FastAPI(
    title="ML Model Serving API",
    description="Classification API for the Iris dataset.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Request failed: %s %s",
            request.method,
            request.url.path,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "method=%s path=%s status=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled error on %s",
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
    }


@app.get("/")
def root():
    return {
        "message": "ML Model Serving API",
        "docs": "/docs",
        "predict_endpoint": "POST /predict",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        values = np.asarray(payload.features, dtype=float)

        if values.shape != (4,):
            raise HTTPException(
                status_code=422,
                detail="Exactly four Iris measurements are required.",
            )

        if not np.isfinite(values).all():
            raise HTTPException(
                status_code=422,
                detail="All feature values must be finite.",
            )

        if (values < 0).any():
            raise HTTPException(
                status_code=422,
                detail="Iris measurements must be non-negative.",
            )

        input_data = values.reshape(1, -1)

        predicted_label = MODEL.predict(input_data)[0]
        class_labels = MODEL.classes_
        probability_values = MODEL.predict_proba(input_data)[0]

        # Convert NumPy scalar labels into ordinary Python values.
        predicted_label = (
            predicted_label.item()
            if isinstance(predicted_label, np.generic)
            else predicted_label
        )

        def species_name(label):
            label = label.item() if isinstance(label, np.generic) else label
            return LABEL_MAP.get(label, str(label))

        probabilities = {
            species_name(label): round(float(probability), 6)
            for label, probability in zip(
                class_labels, probability_values
            )
        }

        return PredictResponse(
            prediction=species_name(predicted_label),
            probabilities=probabilities,
            model_version=MODEL_VERSION,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc
