"""Static tests for Docker and load-testing configuration files.

Validates Dockerfile.onnx, docker-compose.onnx.yml, and run_series.ps1
without invoking Docker or external dependencies.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_onnx_static() -> None:
    """Dockerfile.onnx non-comment lines exclude heavy ML libs and include ONNX setup."""
    dockerfile_path = ROOT / "Dockerfile.onnx"
    assert dockerfile_path.is_file(), f"Missing file: {dockerfile_path}"

    lines = dockerfile_path.read_text(encoding="utf-8").splitlines()
    non_comment_lines = [
        line.strip() for line in lines if line.strip() and not line.strip().startswith("#")
    ]
    non_comment_text = "\n".join(non_comment_lines).lower()

    # Must not contain heavy ML libraries in non-comment lines
    for forbidden in ("scikit-learn", "joblib", "pandas", "scipy"):
        assert forbidden not in non_comment_text, (
            f"Found forbidden dependency {forbidden!r} in non-comment lines of Dockerfile.onnx"
        )

    # Must contain onnxruntime pinned install and STRESS_BACKEND=onnx ENV
    raw_non_comment = "\n".join(non_comment_lines)
    assert "onnxruntime==" in raw_non_comment, (
        "Dockerfile.onnx must contain 'onnxruntime==' in non-comment lines"
    )
    assert "STRESS_BACKEND=onnx" in raw_non_comment, (
        "Dockerfile.onnx must contain 'STRESS_BACKEND=onnx' in non-comment lines"
    )


def test_docker_compose_onnx_static() -> None:
    """docker-compose.onnx.yml specifies Dockerfile.onnx, onnx image, and onnx backend."""
    compose_path = ROOT / "docker-compose.onnx.yml"
    assert compose_path.is_file(), f"Missing file: {compose_path}"

    text = compose_path.read_text(encoding="utf-8")
    assert "Dockerfile.onnx" in text, "docker-compose.onnx.yml must contain 'Dockerfile.onnx'"
    assert "stress-api:onnx" in text, "docker-compose.onnx.yml must contain 'stress-api:onnx'"
    assert "STRESS_BACKEND: onnx" in text, (
        "docker-compose.onnx.yml must contain 'STRESS_BACKEND: onnx'"
    )


def test_run_series_param_block() -> None:
    """run_series.ps1 has exact param block for output directory."""
    script_path = ROOT / "loadtest" / "run_series.ps1"
    assert script_path.is_file(), f"Missing file: {script_path}"

    text = script_path.read_text(encoding="utf-8")
    expected = 'param([string]$Out = "docs\\results\\loadtest")'
    assert expected in text, f"run_series.ps1 must contain exact string: {expected}"
