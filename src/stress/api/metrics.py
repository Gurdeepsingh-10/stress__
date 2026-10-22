"""Prometheus metrics instrumentation for the stress detection API."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Explicit histogram latency buckets for /predict inference (in seconds):
# 0.5 ms, 1 ms, 2.5 ms, 5 ms, 10 ms, 25 ms, 50 ms, 100 ms, 250 ms, 500 ms, 1.0 s
PREDICT_LATENCY_BUCKETS: tuple[float, ...] = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)


@dataclass(frozen=True)
class AppMetrics:
    """Container holding application metrics bound to an isolated CollectorRegistry."""

    registry: CollectorRegistry
    http_requests_total: Counter
    predict_latency_seconds: Histogram
    predictions_total: Counter
    model_loaded: Gauge


def init_metrics() -> AppMetrics:
    """Initialize an isolated per-application Prometheus CollectorRegistry and metrics.

    Returns:
        AppMetrics instance containing the collectors and isolated registry.
    """
    registry = CollectorRegistry(auto_describe=True)

    http_requests_total = Counter(
        "stress_http_requests_total",
        "Total number of HTTP requests processed.",
        labelnames=["route", "method", "status"],
        registry=registry,
    )
    predict_latency_seconds = Histogram(
        "stress_predict_latency_seconds",
        "Latency of /predict endpoint inference in seconds.",
        buckets=PREDICT_LATENCY_BUCKETS,
        registry=registry,
    )
    predictions_total = Counter(
        "stress_predictions_total",
        "Total number of stress predictions produced by predicted label.",
        labelnames=["label"],
        registry=registry,
    )
    model_loaded = Gauge(
        "stress_model_loaded",
        "Indicates whether model artifact is loaded (1) or not (0).",
        registry=registry,
    )
    model_loaded.set(0.0)

    return AppMetrics(
        registry=registry,
        http_requests_total=http_requests_total,
        predict_latency_seconds=predict_latency_seconds,
        predictions_total=predictions_total,
        model_loaded=model_loaded,
    )
