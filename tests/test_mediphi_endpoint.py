"""
Integration test for the MediPhi HuggingFace Inference Endpoint.

Validates:
1. The endpoint is reachable and returns HTTP 200.
2. The response contains a valid chat completion with non-empty content.
3. The LangChain create_llm() factory correctly routes to the endpoint.

Run:
    pytest tests/test_mediphi_endpoint.py -v
    # or directly:
    python tests/test_mediphi_endpoint.py
"""

import os
import sys
import pytest
from pathlib import Path

# Ensure project root is on sys.path so src.* imports resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

MEDIPHI_ENDPOINT = os.getenv("MEDIPHI_ENDPOINT_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

requires_endpoint = pytest.mark.skipif(
    not MEDIPHI_ENDPOINT or "your_mediphi" in MEDIPHI_ENDPOINT,
    reason="MEDIPHI_ENDPOINT_URL not configured",
)
requires_token = pytest.mark.skipif(
    not HF_TOKEN,
    reason="HF_TOKEN not set",
)

TEST_PROMPT = "What are the key clinical findings when a patient presents with chest pain and dyspnea?"
SYSTEM_PROMPT = "You are a clinical NLP assistant. Answer concisely."


# ---------------------------------------------------------------------------
# Test 1 – raw HTTP call (mirrors the curl example, no LangChain)
# ---------------------------------------------------------------------------

@requires_endpoint
@requires_token
def test_raw_http_chat_completion():
    """Direct HTTP POST to /v1/chat/completions — validates endpoint is live."""
    import requests  # type: ignore

    url = f"{MEDIPHI_ENDPOINT}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "microsoft/MediPhi",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TEST_PROMPT},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. Body: {response.text}"
    )

    data = response.json()
    assert "choices" in data, f"'choices' key missing from response: {data}"
    assert len(data["choices"]) > 0, "Response has empty 'choices'"

    content = data["choices"][0]["message"]["content"]
    assert content and content.strip(), "Response content is empty"

    print(f"\n[raw HTTP] MediPhi response ({len(content)} chars):\n{content[:300]}")


# ---------------------------------------------------------------------------
# Test 2 – LangChain factory (create_llm + MODELS_REGISTRY)
# ---------------------------------------------------------------------------

@requires_endpoint
@requires_token
def test_langchain_create_llm_mediphi():
    """Validates that create_llm() builds a working ChatOpenAI for MediPhi."""
    from src.common.model_provider import create_llm, MODELS_REGISTRY
    from langchain_core.messages import HumanMessage, SystemMessage

    assert "mediphi" in MODELS_REGISTRY, "'mediphi' not found in MODELS_REGISTRY"

    llm = create_llm(MODELS_REGISTRY["mediphi"])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=TEST_PROMPT),
    ]

    response = llm.invoke(messages)

    assert response.content and response.content.strip(), (
        "LangChain response content is empty"
    )

    print(f"\n[LangChain] MediPhi response ({len(response.content)} chars):\n{response.content[:300]}")


# ---------------------------------------------------------------------------
# Test 3 – model_id is passed correctly (not hardcoded "tgi")
# ---------------------------------------------------------------------------

def test_model_config_model_id_is_not_tgi():
    """Ensures create_llm uses config.model_id, not the old hardcoded 'tgi'."""
    from src.common.model_provider import MODELS_REGISTRY

    config = MODELS_REGISTRY["mediphi"]
    assert config.model_id == "microsoft/MediPhi", (
        f"Expected 'microsoft/MediPhi', got '{config.model_id}'"
    )


# ---------------------------------------------------------------------------
# Standalone runner (python tests/test_mediphi_endpoint.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MediPhi Endpoint Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    def run(name, fn):
        global passed, failed
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}\n        {exc}")
            failed += 1

    run("model_id not 'tgi'", test_model_config_model_id_is_not_tgi)

    if not MEDIPHI_ENDPOINT or "your_mediphi" in MEDIPHI_ENDPOINT:
        print("\n  SKIP  raw HTTP test  (MEDIPHI_ENDPOINT_URL not set)")
        print("  SKIP  LangChain test (MEDIPHI_ENDPOINT_URL not set)")
    elif not HF_TOKEN:
        print("\n  SKIP  raw HTTP test  (HF_TOKEN not set)")
        print("  SKIP  LangChain test (HF_TOKEN not set)")
    else:
        run("raw HTTP chat completion", test_raw_http_chat_completion)
        run("LangChain create_llm mediphi", test_langchain_create_llm_mediphi)

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
