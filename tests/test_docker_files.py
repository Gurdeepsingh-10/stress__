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


def _extract_pip_install_block(content: str) -> str:
    """Extract the multi-line pip install block starting with RUN pip install --no-cache-dir \\."""
    lines = content.splitlines()
    block: list[str] = []
    capturing = False
    for line in lines:
        if line.startswith("RUN pip install --no-cache-dir \\"):
            capturing = True
            block.append(line)
        elif capturing:
            block.append(line)
            if not line.rstrip().endswith("\\"):
                break
    return "\n".join(block)


def test_dockerfile_deploy_static() -> None:
    """Dockerfile.deploy static checks: exclusions, COPY, PORT, and identical pip install."""
    deploy_path = ROOT / "Dockerfile.deploy"
    onnx_path = ROOT / "Dockerfile.onnx"
    assert deploy_path.is_file(), f"Missing file: {deploy_path}"
    assert onnx_path.is_file(), f"Missing file: {onnx_path}"

    deploy_text = deploy_path.read_text(encoding="utf-8")
    onnx_text = onnx_path.read_text(encoding="utf-8")

    lines = deploy_text.splitlines()
    non_comment_lines = [
        line.strip() for line in lines if line.strip() and not line.strip().startswith("#")
    ]
    non_comment_text = "\n".join(non_comment_lines).lower()

    # Must not contain heavy ML libraries in non-comment lines
    for forbidden in ("scikit-learn", "joblib", "pandas", "scipy"):
        assert forbidden not in non_comment_text, (
            f"Found forbidden dependency {forbidden!r} in non-comment lines of Dockerfile.deploy"
        )

    # Must contain COPY deploy/models, ${PORT:-8000}, and STRESS_BACKEND=onnx
    raw_non_comment = "\n".join(non_comment_lines)
    assert "COPY --chown=10001:10001 deploy/models/" in raw_non_comment, (
        "Dockerfile.deploy must contain 'COPY --chown=10001:10001 deploy/models/' "
        "in non-comment lines"
    )
    assert "${PORT:-8000}" in raw_non_comment, (
        "Dockerfile.deploy must contain '${PORT:-8000}' in non-comment lines"
    )
    assert "STRESS_BACKEND=onnx" in raw_non_comment, (
        "Dockerfile.deploy must contain 'STRESS_BACKEND=onnx' in non-comment lines"
    )

    # Multi-line pip install block must be byte-identical
    deploy_pip = _extract_pip_install_block(deploy_text)
    onnx_pip = _extract_pip_install_block(onnx_text)
    assert deploy_pip, "Failed to extract pip install block from Dockerfile.deploy"
    assert deploy_pip == onnx_pip, (
        "The multi-line pip install block in Dockerfile.deploy must be identical to Dockerfile.onnx"
    )
