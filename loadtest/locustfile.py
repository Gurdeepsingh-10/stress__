"""Locust load test for the stress detection API /predict endpoint.

Cycles through the first N feature rows from CSV with zero think time
and validates HTTP status and response payload bounds.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from locust import HttpUser, constant, task

from stress.api.payload_builder import load_payloads

_DEFAULT_CSV: Path = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "features_phase2.csv"
)


def _init_payload_cycle() -> Iterator[dict[str, Any]]:
    """Load payloads from CSV and return an infinite iterator cycling through them."""
    csv_env: str = os.getenv("STRESS_CSV", "")
    csv_path: Path = Path(csv_env) if csv_env else _DEFAULT_CSV

    n_env: str = os.getenv("STRESS_PAYLOAD_N", "0")
    try:
        n: int = int(n_env)
    except ValueError:
        n = 0

    payloads: list[dict[str, Any]] = load_payloads(csv_path, n=n if n > 0 else None)
    if not payloads:
        raise ValueError(f"No valid payloads loaded from {csv_path}")
    return itertools.cycle(payloads)


# Shared cycle across all virtual users in the runner process
_PAYLOAD_CYCLE: Iterator[dict[str, Any]] = _init_payload_cycle()


class StressUser(HttpUser):
    """Virtual user driving high-throughput stress inference load against /predict."""

    # Zero think time maximizes client-side throughput
    wait_time = constant(0)

    @task
    def predict(self) -> None:
        """Send a single inference request to /predict and validate response bounds."""
        payload: dict[str, Any] = next(_PAYLOAD_CYCLE)
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Unexpected status code {resp.status_code}: {resp.text}")
                return

            try:
                data: Any = resp.json()
            except Exception as exc:  # noqa: BLE001
                resp.failure(f"Failed to decode response as JSON: {exc}")
                return

            if not isinstance(data, dict):
                resp.failure("Response body is not a JSON object")
                return

            prob: Any = data.get("stress_probability")
            if (
                prob is None
                or not isinstance(prob, (int, float))
                or not (0.0 <= float(prob) <= 1.0)
            ):
                resp.failure(f"Invalid stress_probability in response: {prob}")
                return

            resp.success()
