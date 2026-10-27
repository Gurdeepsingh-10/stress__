"""Static tests for Grafana dashboard, provisioning configs, and compose override."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = ROOT / "monitoring" / "grafana" / "dashboards" / "stress-api.json"
DATASOURCE_PATH = (
    ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
)
DASHBOARDS_PROV_PATH = (
    ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
)
COMPOSE_PATH = ROOT / "docker-compose.grafana.yml"
METRICS_PY_PATH = ROOT / "src" / "stress" / "api" / "metrics.py"


def test_dashboard_json_structure() -> None:
    """(i) Dashboard JSON parses, uid is 'stress-api', and has at least 4 panels."""
    assert DASHBOARD_PATH.is_file(), f"Missing file: {DASHBOARD_PATH}"
    data = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))

    assert data.get("uid") == "stress-api", f"Expected uid 'stress-api', got {data.get('uid')!r}"
    assert data.get("title") == "Stress API", (
        f"Expected title 'Stress API', got {data.get('title')!r}"
    )

    panels = data.get("panels", [])
    assert len(panels) >= 4, f"Dashboard must have at least 4 panels, found {len(panels)}"


def test_dashboard_metric_names_exist_in_metrics_py() -> None:
    """(ii) Every metric name in any panel expr exists in src/stress/api/metrics.py source."""
    assert DASHBOARD_PATH.is_file(), f"Missing file: {DASHBOARD_PATH}"
    assert METRICS_PY_PATH.is_file(), f"Missing file: {METRICS_PY_PATH}"

    data = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    metrics_source = METRICS_PY_PATH.read_text(encoding="utf-8")

    expr_strings: list[str] = []
    for panel in data.get("panels", []):
        if "expr" in panel:
            expr_strings.append(panel["expr"])
        for target in panel.get("targets", []):
            if "expr" in target:
                expr_strings.append(target["expr"])

    assert expr_strings, "No target expressions found in dashboard panels"

    metric_pattern = re.compile(r"\bstress_[a-zA-Z0-9_]+\b")
    found_metrics: set[str] = set()

    for expr in expr_strings:
        for match in metric_pattern.finditer(expr):
            raw_metric = match.group(0)
            # Strip Prometheus exposition suffixes: _bucket, _sum, _count
            base_metric = re.sub(r"_(bucket|sum|count)$", "", raw_metric)
            found_metrics.add(base_metric)

    assert found_metrics, "No 'stress_' metrics found in panel expressions"

    for metric in sorted(found_metrics):
        quoted_double = f'"{metric}"'
        quoted_single = f"'{metric}'"
        assert quoted_double in metrics_source or quoted_single in metrics_source, (
            f"Metric {metric!r} from dashboard expression is not defined in {METRICS_PY_PATH}"
        )


def test_dashboard_and_target_datasource_uid_equals_provisioning() -> None:
    """(iii) Every panel and target datasource uid equals the uid in prometheus.yml."""
    assert DATASOURCE_PATH.is_file(), f"Missing file: {DATASOURCE_PATH}"
    assert DASHBOARD_PATH.is_file(), f"Missing file: {DASHBOARD_PATH}"

    ds_config = yaml.safe_load(DATASOURCE_PATH.read_text(encoding="utf-8"))
    assert "datasources" in ds_config and ds_config["datasources"], "Missing datasources list"
    expected_uid = ds_config["datasources"][0].get("uid")
    assert expected_uid == "prometheus-stress", (
        f"Expected 'prometheus-stress', got {expected_uid!r}"
    )

    data = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    panels = data.get("panels", [])
    assert panels, "No panels in dashboard"

    for idx, panel in enumerate(panels):
        panel_ds = panel.get("datasource")
        if isinstance(panel_ds, dict):
            actual_uid = panel_ds.get("uid")
        else:
            actual_uid = panel_ds
        assert actual_uid == expected_uid, (
            f"Panel {idx} ('{panel.get('title')}') datasource uid {actual_uid!r} "
            f"does not match expected {expected_uid!r}"
        )

        for t_idx, target in enumerate(panel.get("targets", [])):
            target_ds = target.get("datasource")
            if isinstance(target_ds, dict):
                t_uid = target_ds.get("uid")
            else:
                t_uid = target_ds
            assert t_uid == expected_uid, (
                f"Panel {idx} target {t_idx} datasource uid {t_uid!r} "
                f"does not match expected {expected_uid!r}"
            )


def test_docker_compose_grafana_static() -> None:
    """(iv) Compose override: image pinned, 127.0.0.1 port, required password, prometheus dep."""
    assert COMPOSE_PATH.is_file(), f"Missing file: {COMPOSE_PATH}"
    compose_text = COMPOSE_PATH.read_text(encoding="utf-8")

    # Image is pinned (contains a version digit tag, not ':latest' and not untagged)
    image_match = re.search(r"image:\s*([^\s]+)", compose_text)
    assert image_match, "Missing image declaration in docker-compose.grafana.yml"
    image_str = image_match.group(1).strip("\"'")
    assert ":" in image_str, f"Image {image_str!r} is untagged"
    tag = image_str.split(":", 1)[1]
    assert tag != "latest", "Image tag must not be 'latest'"
    assert any(c.isdigit() for c in tag), f"Image tag {tag!r} must contain a version digit"

    # Port binding starts with 127.0.0.1
    port_lines = [
        line.strip().lstrip("-").strip().strip("\"'")
        for line in compose_text.splitlines()
        if "3000:3000" in line
    ]
    assert port_lines, "Expected port mapping containing 3000:3000"
    for port_str in port_lines:
        assert port_str.startswith("127.0.0.1:"), (
            f"Port binding {port_str!r} must start with '127.0.0.1:'"
        )

    # GRAFANA_ADMIN_PASSWORD uses the ':?' required form
    assert "${GRAFANA_ADMIN_PASSWORD:?" in compose_text, (
        "GF_SECURITY_ADMIN_PASSWORD must use the '${GRAFANA_ADMIN_PASSWORD:?...}' required form"
    )

    # A prometheus dependency is present
    dep_match = re.search(
        r"depends_on:\s*(\n\s*-\s*prometheus|\n\s*prometheus:)",
        compose_text,
    )
    assert dep_match, "docker-compose.grafana.yml must declare depends_on the prometheus service"
