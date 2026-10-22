# Multi-stage build for stress detection FastAPI service.
# Base image pinned to python:3.11-slim (no latest tag).

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Create isolated virtual environment in /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to modern standard in the isolated venv
RUN pip install --no-cache-dir --upgrade pip

# Install pinned runtime dependencies matching the model training environment.
# Note: mlflow, dev, and test tools are deliberately excluded to minimize image size.
RUN pip install --no-cache-dir \
    scikit-learn==1.9.1 \
    numpy==2.4.6 \
    pandas==3.0.6 \
    scipy==1.17.1 \
    joblib==1.6.0 \
    fastapi==0.143.0 \
    "uvicorn[standard]==0.54.0" \
    pydantic==2.14.1 \
    prometheus-client==0.26.0

# Install project package (stress) from source.
# Justification: --no-deps prevents pulling in unpinned optional dependencies
# (like mlflow and its heavy dependencies declared in pyproject.toml dependencies).
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Copy only the compiled virtual environment from builder stage (no compilers or build tools)
COPY --from=builder /opt/venv /opt/venv

# Configure runtime environment variables
# STRESS_MODEL_DIR points to the mount destination for model artifacts.
ENV STRESS_MODEL_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create fixed UID non-root system user with no login shell for security.
# Pre-create /app/models directory and assign ownership so bind mounts are readable.
RUN useradd -u 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

# Run as non-root user
USER 10001

EXPOSE 8000

# Health check using Python standard library urllib.request to avoid external tools (curl/wget).
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).getcode() == 200 else 1)"]

# Start FastAPI application using factory pattern in exec form
CMD ["uvicorn", "stress.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
