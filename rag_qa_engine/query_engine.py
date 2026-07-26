"""Command-line client for the SmartOps RAG API."""

import os

import requests

API_URL = os.getenv("SMARTOPS_API_URL", "http://127.0.0.1:8000")


def ask_smartops(user_question: str) -> dict:
    """Send a policy question to the same RAG endpoint used by the dashboard."""
    response = requests.post(
        f"{API_URL.rstrip('/')}/api/v1/query", json={"query": user_question}, timeout=20
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    result = ask_smartops("What is the target response time for Tier-1 customer support tickets?")
    print(result["answer"])
    print("Sources:", result["sources"])
