"""Tests for Prometheus metrics instrumentation in stress.api.main and stress.api.metrics."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from stress.api.main import create_app
from tests.test_model_loader import _save_valid_artifact_bundle, _valid_request_dict


def get_sample_value(
    metrics_text: str,
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    """Extract a sample value matching metric name and optional label set."""
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name == metric_name:
                if labels is None or sample.labels == labels:
                    return float(sample.value)
    return None


def test_predict_success_increments_request_and_prediction_counter(tmp_path: Path) -> None:
    """After one successful POST /predict, request and prediction counters increment."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        res = client.post("/predict", json=_valid_request_dict())
        assert res.status_code == 200
        predicted_label = str(res.json()["predicted_label"])

        metrics_res = client.get("/metrics")
        assert metrics_res.status_code == 200
        text = metrics_res.text

        req_count = get_sample_value(
            text,
            "stress_http_requests_total",
            {"route": "/predict", "method": "POST", "status": "200"},
        )
        assert req_count == 1.0

        pred_count = get_sample_value(
            text,
            "stress_predictions_total",
            {"label": predicted_label},
        )
        assert pred_count == 1.0


def test_predict_validation_error_records_422_without_prediction_increment(
    tmp_path: Path,
) -> None:
    """After a bad POST /predict (422), 422 is counted and predictions counter is unchanged."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        bad_payload = dict(_valid_request_dict())
        del bad_payload["temp_mean"]  # missing required field produces 422

        res = client.post("/predict", json=bad_payload)
        assert res.status_code == 422

        metrics_res = client.get("/metrics")
        text = metrics_res.text

        req_count_422 = get_sample_value(
            text,
            "stress_http_requests_total",
            {"route": "/predict", "method": "POST", "status": "422"},
        )
        assert req_count_422 == 1.0

        pred_0 = get_sample_value(text, "stress_predictions_total", {"label": "0"})
        pred_1 = get_sample_value(text, "stress_predictions_total", {"label": "1"})
        assert pred_0 is None
        assert pred_1 is None


def test_nonexistent_route_records_unmatched_and_omits_raw_path(tmp_path: Path) -> None:
    """GET /nonexistent-xyz is counted as route='unmatched' with status='404' without raw path."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        res = client.get("/nonexistent-xyz")
        assert res.status_code == 404

        metrics_res = client.get("/metrics")
        text = metrics_res.text

        req_count = get_sample_value(
            text,
            "stress_http_requests_total",
            {"route": "unmatched", "method": "GET", "status": "404"},
        )
        assert req_count == 1.0
        assert "nonexistent-xyz" not in text


def test_metrics_endpoint_exposition_and_model_loaded_gauge(tmp_path: Path) -> None:
    """GET /metrics returns text/plain exposition containing stress_model_loaded 1.0."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        res = client.get("/metrics")
        assert res.status_code == 200
        assert "text/plain" in res.headers["content-type"]

        model_loaded = get_sample_value(res.text, "stress_model_loaded")
        assert model_loaded == 1.0


def test_independent_collector_registries_between_app_instances(tmp_path: Path) -> None:
    """Two create_app() calls do not raise duplicate-series errors and have independent counters."""
    _save_valid_artifact_bundle(tmp_path)
    app1 = create_app(tmp_path)
    app2 = create_app(tmp_path)

    with TestClient(app1) as client1, TestClient(app2) as client2:
        # Request on client1 only
        res1 = client1.post("/predict", json=_valid_request_dict())
        assert res1.status_code == 200

        text1 = client1.get("/metrics").text
        text2 = client2.get("/metrics").text

        count1 = get_sample_value(
            text1,
            "stress_http_requests_total",
            {"route": "/predict", "method": "POST", "status": "200"},
        )
        count2 = get_sample_value(
            text2,
            "stress_http_requests_total",
            {"route": "/predict", "method": "POST", "status": "200"},
        )

        assert count1 == 1.0
        assert count2 is None


def test_metrics_endpoint_scrapes_do_not_increment_request_counter(tmp_path: Path) -> None:
    """Scraping /metrics does not increment stress_http_requests_total."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        for _ in range(5):
            res = client.get("/metrics")
            assert res.status_code == 200

        text = client.get("/metrics").text
        sample = get_sample_value(
            text,
            "stress_http_requests_total",
            {"route": "/metrics", "method": "GET", "status": "200"},
        )
        assert sample is None
