"""FastAPI application for the stress detection research demo service."""

from __future__ import annotations

import math
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from stress.api.metrics import AppMetrics, init_metrics
from stress.api.model_loader import load_model, predict_stress_probability
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

_ALLOWED_BACKENDS: frozenset[str] = frozenset({"sklearn", "onnx"})


def decide_label(p: float, threshold: float = THRESHOLD) -> Literal[0, 1]:
    """Classify stress probability into binary label using strict inequality (>).

    Args:
        p: Predicted stress probability in [0.0, 1.0].
        threshold: Decision threshold. Defaults to THRESHOLD (0.5).

    Returns:
        1 if p > threshold else 0.
    """
    return 1 if p > threshold else 0


def create_app(
    model_dir: Path | str | None = None,
    backend: str | None = None,
) -> FastAPI:
    """Create and configure FastAPI application with lifespan model loading and metrics.

    Args:
        model_dir: Directory containing model artifacts. If None, defaults to
            STRESS_MODEL_DIR environment variable or 'models'.
        backend: Inference backend to use: 'sklearn' or 'onnx'. If None, resolved
            from STRESS_BACKEND environment variable, then defaults to 'sklearn'.

    Returns:
        Configured FastAPI application instance.

    Raises:
        ValueError: If backend is not one of the allowed values.
    """
    # Resolve backend: argument > env > default
    resolved_backend: str = (
        backend if backend is not None else os.environ.get("STRESS_BACKEND", "sklearn")
    )
    if resolved_backend not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"Invalid backend {resolved_backend!r}. Allowed values: {sorted(_ALLOWED_BACKENDS)}"
        )

    if resolved_backend == "onnx":
        from stress.api.onnx_backend import onnx_stress_probability

        onnx_predict = onnx_stress_probability
    else:
        onnx_predict = None

    metrics = init_metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if resolved_backend == "onnx":
            from stress.api.onnx_loader import load_onnx_model

            app.state.model = load_onnx_model(model_dir)
        else:
            app.state.model = load_model(model_dir)
        app.state.backend = resolved_backend
        app.state.metrics.model_loaded.set(1.0)
        try:
            yield
        finally:
            app.state.metrics.model_loaded.set(0.0)

    app = FastAPI(
        title="Stress Detection Research Demo",
        description=APP_DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.metrics = metrics
    app.state.metrics_registry = metrics.registry

    @app.middleware("http")
    async def metrics_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            route_path = route.path if route and hasattr(route, "path") else "unmatched"
            if route_path != "/metrics":
                app_metrics: AppMetrics = request.app.state.metrics
                app_metrics.http_requests_total.labels(
                    route=route_path,
                    method=request.method,
                    status=str(status_code),
                ).inc()

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
        loaded = request.app.state.model
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
            backend=request.app.state.backend,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        start_time = time.perf_counter()
        loaded = request.app.state.model
        if onnx_predict is not None:
            p = onnx_predict(loaded.session, payload, loaded.metadata["features"])
        else:
            p = predict_stress_probability(loaded, payload)
        label = decide_label(p, threshold=THRESHOLD)
        duration = time.perf_counter() - start_time

        app_metrics: AppMetrics = request.app.state.metrics
        app_metrics.predict_latency_seconds.observe(duration)
        app_metrics.predictions_total.labels(label=str(label)).inc()

        return PredictResponse(
            stress_probability=p,
            predicted_label=label,
            threshold=THRESHOLD,
            model_version=str(loaded.metadata["model_version"]),
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint(request: Request) -> Response:
        data = generate_latest(request.app.state.metrics_registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
