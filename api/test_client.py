"""Manual smoke-test client for a running SmartOps API instance.

Run this file directly after starting Uvicorn; it is intentionally not a pytest test.
"""

import os

import requests

BASE_URL = os.getenv("SMARTOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")


def run_smoke_test() -> dict:
    """Submit a representative customer profile and return the API response."""
    response = requests.post(
        f"{BASE_URL}/predict/CU-9982",
        json={"recency": 2, "frequency": 25, "monetary_value": 1450.50, "refund_rate": 0.05},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print(run_smoke_test())
