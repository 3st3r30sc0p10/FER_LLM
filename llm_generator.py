"""
Module 4 & 5: Sentence Generator (LLM)
Supports Ollama (local) and OpenAI API. Emotion defines the constraint; LLM generates text.
"""

import os
import requests
from typing import Optional

# Optional OpenAI client (only used when backend is "openai")
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def generate_ollama(prompt: str, model: str = "mistral", stream: bool = False) -> str:
    """Generate text via local Ollama. Requires Ollama running (e.g. ollama run mistral)."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


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


def generate(prompt: str, backend: str = "ollama", **kwargs) -> str:
    """
    Generate poetic text from prompt.
    backend: "ollama" | "openai"
    Ollama kwargs: model (default mistral), stream (default False).
    OpenAI kwargs: model (default gpt-4o-mini).
    """
    if backend == "ollama":
        return generate_ollama(
            prompt,
            model=kwargs.get("model", "mistral"),
            stream=kwargs.get("stream", False),
        )
    if backend == "openai":
        return generate_openai(prompt, model=kwargs.get("model", "gpt-4o-mini"))
    raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'openai'.")
