"""
Module 4 & 5: Sentence Generator (LLM)
Supports DukeGPT (Duke LiteLLM) and OpenAI API. Emotion defines the constraint; LLM generates text.

DukeGPT architecture: this app → proxy server → Duke API. The app talks only to the proxy
(default http://localhost:3001). The proxy holds the Duke URL and API key and forwards to Duke.
Only the machine running the proxy needs to reach Duke's API (e.g. Duke VPN on that machine).
"""

import os
import requests
from typing import Optional

# Default: proxy URL. App never talks to Duke directly; proxy does.
# Override with DUKEGPT_API_URL (e.g. http://localhost:3001 or a deployed proxy host).
PROXY_DEFAULT_URL = "http://localhost:3001"
_raw_url = os.environ.get("DUKEGPT_API_URL", PROXY_DEFAULT_URL)
DUKEGPT_API_URL = PROXY_DEFAULT_URL if "your-host" in _raw_url else _raw_url
# Only used when calling Duke directly (not via proxy); proxy has its own key.
DUKEGPT_API_KEY = os.environ.get("DUKEGPT_API_KEY", "sk-dwAYbKw4KalzudSkQVcOWg")
DUKEGPT_DEFAULT_MODEL = "GPT 4.1"


def get_dukegpt_url(override: Optional[str] = None) -> str:
    """Return the Duke LLM endpoint URL (proxy or direct)."""
    url = override or DUKEGPT_API_URL
    url = url.rstrip("/")
    # Full chat/completions endpoint = direct Duke: use as-is. Otherwise = proxy: append /proxy/llm
    if "chat/completions" in url:
        return url
    if not url.endswith("proxy/llm"):
        url = url + "/proxy/llm"
    return url


def _is_direct_duke_url(url: str) -> bool:
    """True if url is Duke API directly (we send API key). False for proxy (proxy has key)."""
    return "chat/completions" in url

# Optional OpenAI client (only used when backend is "openai")
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def generate_dukegpt(prompt: str, model: str = DUKEGPT_DEFAULT_MODEL, base_url: Optional[str] = None) -> str:
    """Generate via proxy (default) or direct Duke API. Proxy holds Duke URL and key; app sends none."""
    url = get_dukegpt_url(base_url)
    headers = {"Content-Type": "application/json"}
    # Only send API key when calling Duke directly; proxy adds its own key when it calls Duke.
    if _is_direct_duke_url(url) and DUKEGPT_API_KEY:
        headers["Authorization"] = f"Bearer {DUKEGPT_API_KEY}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a poetic generator. Output only the requested sentence, no explanation or preamble.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"LLM endpoint unreachable at {url}. "
            "Start the proxy first: cd llm-proxy && npm install && npm start"
        ) from e
    data = response.json()
    if not data.get("choices") or not data["choices"][0].get("message"):
        raise ValueError("Invalid response format from DukeGPT API")
    text = data["choices"][0]["message"].get("content", "")
    return (text or "").strip()


def generate_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Generate text via OpenAI API. Requires OPENAI_API_KEY in environment."""
    if OpenAI is None:
        raise ImportError("openai package required for OpenAI backend. pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    text = response.choices[0].message.content
    return (text or "").strip()


def generate(prompt: str, backend: str = "dukegpt", **kwargs) -> str:
    """
    Generate poetic text from prompt.
    backend: "dukegpt" | "openai"
    DukeGPT kwargs: model (default GPT 4.1), base_url. Uses DUKEGPT_API_URL, DUKEGPT_API_KEY env.
    OpenAI kwargs: model (default gpt-4o-mini).
    """
    if backend == "openai":
        return generate_openai(prompt, model=kwargs.get("model", "gpt-4o-mini"))
    if backend == "dukegpt":
        return generate_dukegpt(
            prompt,
            model=kwargs.get("model", DUKEGPT_DEFAULT_MODEL),
            base_url=kwargs.get("dukegpt_url"),
        )
    raise ValueError(f"Unknown backend: {backend}. Use 'dukegpt' or 'openai'.")
