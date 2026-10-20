"""FastAPI application for the stress detection research demo service."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from stress.api.model_loader import LoadedModel, load_model, predict_stress_probability
from stress.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

THRESHOLD: float = 0.5

APP_DESCRIPTION: str = (
    "Research demo for acute stress detection based on wrist-wearable "
    "physiological data (WESAD).\n\n"
    "RESEARCH DEMO — NOT A MEDICAL DEVICE.\n"
    "This service is provided for scientific research and benchmark evaluation only. "
    "It is not intended, certified, or cleared for medical diagnosis, patient monitoring, "
    "or clinical decision support."
)


def decide_label(p: float, threshold: float = THRESHOLD) -> Literal[0, 1]:
    """Classify stress probability into binary label using strict inequality (>).

    Args:
        p: Predicted stress probability in [0.0, 1.0].
        threshold: Decision threshold. Defaults to THRESHOLD (0.5).

    Returns:
        1 if p > threshold else 0.
    """
    return 1 if p > threshold else 0


def create_app(model_dir: Path | str | None = None) -> FastAPI:
    """Create and configure FastAPI application with lifespan model loading.

    Args:
        model_dir: Directory containing model artifacts. If None, defaults to
            STRESS_MODEL_DIR environment variable or 'models'.

    Returns:
        Configured FastAPI application instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = load_model(model_dir)
        yield

    app = FastAPI(
        title="Stress Detection Research Demo",
        description=APP_DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = []
        for err in exc.errors():
            err_dict = dict(err)
            input_val = err_dict.get("input")
            if isinstance(input_val, float) and (math.isnan(input_val) or math.isinf(input_val)):
                err_dict["input"] = str(input_val)
            errors.append(err_dict)
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = getattr(request.app.state, "model", None)
        return HealthResponse(status="ok", model_loaded=loaded is not None)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        loaded: LoadedModel = request.app.state.model
        meta = loaded.metadata
        return ModelInfoResponse(
            model_name=str(meta["model_name"]),
            model_version=str(meta["model_version"]),
            target=str(meta["target"]),
            trained_at_utc=str(meta["trained_at_utc"]),
            git_commit=str(meta["git_commit"]),
            git_dirty=bool(meta["git_dirty"]) if meta.get("git_dirty") is not None else False,
            features=list(meta["features"]),
            n_features=int(meta["n_features"]),
            training_samples=int(meta["training_samples"]),
            environment={k: str(v) for k, v in meta.get("environment", {}).items()},
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        loaded: LoadedModel = request.app.state.model
        p = predict_stress_probability(loaded, payload)
        label = decide_label(p, threshold=THRESHOLD)
        return PredictResponse(
            stress_probability=p,
            predicted_label=label,
            threshold=THRESHOLD,
            model_version=str(loaded.metadata["model_version"]),
        )

    return app


app = create_app()
