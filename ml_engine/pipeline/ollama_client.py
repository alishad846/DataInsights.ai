"""
ollama_client.py — Lightweight Ollama API wrapper
===================================================
Used ONLY for:
1. Greeting responses in query_engine.py
2. Low-confidence column interpretation in schema_manager.py

NOT used for: analytics, BI, forecasting, or dashboard generation.
The pipeline remains fully deterministic when Ollama is unavailable.
"""

import json
import concurrent.futures
import logging
import urllib.request
import urllib.error

logger = logging.getLogger("system_logger")

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"
OLLAMA_TIMEOUT = 20  # seconds hard limit — enforced via thread so CPU-bound models can't hang pipeline


def _is_available() -> bool:
    """Quick health check — returns True if Ollama is reachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def _generate_blocking(prompt: str, system: str) -> str | None:
    """Internal blocking call — always run inside a thread via generate()."""
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 128,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response = data.get("response", "").strip()

            if not response:
                logger.warning("[Ollama] Empty response from model")
                return None

            error_patterns = [
                "does not support image",
                "cannot read image",
                "not support",
                "vision",
                "multimodal",
            ]
            if any(pattern in response.lower() for pattern in error_patterns):
                logger.warning(
                    f"[Ollama] Model does not support image input - returning fallback"
                )
                return None

            return response
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        if (
            "does not support image" in error_body.lower()
            or "cannot read image" in error_body.lower()
            or e.code == 400
        ):
            logger.warning(
                f"[Ollama] HTTP {e.code} - model does not support image input, skipping"
            )
            return None
        logger.warning(f"[Ollama] HTTP error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        logger.warning(f"[Ollama] Connection error: {e.reason}")
        return None
    except Exception as e:
        logger.warning(f"[Ollama] Unexpected error: {e}")
        return None


def generate(prompt: str, system: str = "") -> str | None:
    """
    Call Ollama with a hard thread timeout so slow CPU inference
    never blocks the ML pipeline. Returns None on any failure.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_generate_blocking, prompt, system)
        try:
            return future.result(timeout=OLLAMA_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"[Ollama] Timed out after {OLLAMA_TIMEOUT}s — falling back to rule-based."
            )
            return None
        except Exception as e:
            logger.warning(f"[Ollama] Request failed: {e}")
            return None


def schema_interpret(columns: list[str]) -> dict | None:
    """
    Ask Ollama to identify business roles (date, sales, customer, product)
    from a list of column names. Returns a dict or None on failure.
    """
    if not _is_available():
        logger.warning("[Ollama] Not available — skipping schema interpretation.")
        return None

    logger.info("[Ollama] Schema interpretation triggered due to low confidence.")

    col_list = json.dumps(columns)
    prompt = (
        f"Dataset columns: {col_list}\n\n"
        "Identify which column is the date column, sales/revenue column, "
        "customer column, and product column (if any).\n"
        "Reply with ONLY minified JSON like:\n"
        '{"date_column":"X","sales_column":"Y","customer_column":"Z","product_column":"W"}\n'
        "If a role has no match, omit that key. No explanation, just JSON."
    )

    raw = generate(prompt)
    if not raw:
        return None

    # Extract first JSON object from response (model sometimes adds text)
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[Ollama] Could not parse schema JSON from response: {e}")
        return None


def greeting_response(user_message: str) -> str | None:
    """
    Generate a friendly assistant reply for greetings.
    Returns None if Ollama is unavailable (caller will use a static fallback).
    """
    if not _is_available():
        logger.warning("[Ollama] Not available — using static greeting fallback.")
        return None

    logger.info("[Ollama] Greeting handler triggered.")

    prompt = (
        f'User said: "{user_message}"\n'
        "Respond as a friendly data analytics assistant for DataInsights.ai. "
        "Be concise (2-3 sentences max). Mention you can answer questions about "
        "their uploaded dataset such as totals, trends, averages, and insights."
    )
    return generate(prompt)
