"""Shared LLM API call utility using stdlib ``urllib``.

Provides a single function :func:`call_chat_completion` for calling any
OpenAI-compatible chat completion endpoint.  Used internally by the
``llm-nli`` similarity backend and by the answer/context relevancy scorers.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_TIMEOUT = 120


def call_chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint.

    Parameters
    ----------
    messages:
        List of ``{"role": ..., "content": ...}`` dicts.
    model:
        Model identifier (e.g. ``"gpt-4o-mini"``).
    api_key:
        Bearer token.  Falls back to the ``OPENAI_API_KEY`` env var.
    base_url:
        API base URL.  Defaults to ``https://api.openai.com/v1``.
    temperature:
        Sampling temperature.  Default ``0.0`` for deterministic output.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    str
        The assistant message content string.

    Raises
    ------
    ValueError
        When no API key is available.
    RuntimeError
        On HTTP errors or connection failures.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "LLM API key is required. Set the OPENAI_API_KEY environment "
            "variable or pass llm_api_key in EvalConfig."
        )

    url = f"{(base_url or _DEFAULT_BASE_URL).rstrip('/')}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM API returned HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API connection failed: {exc.reason}") from exc
