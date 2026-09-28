"""
Unified LLM client supporting multiple providers.
Supports: OpenAI (GPT), Anthropic (Claude), compatible APIs (Ollama local models).
"""

import os
import time
from dataclasses import dataclass, field
from openai import OpenAI

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    display_name: str = ""
    provider: str = "openai"  # "openai" | "anthropic" | "ollama"

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.model


# Pre-configured models
MODELS = {
    # === OpenAI ===
    "gpt-4o-mini": LLMConfig(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        display_name="GPT-4o-mini",
        provider="openai",
    ),
    "gpt-6-astra": LLMConfig(
        model=os.getenv("GPT6_MODEL_ID", "gpt-6-astra"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        display_name="GPT-6 Astra",
        provider="openai",
    ),

    # === Anthropic Claude ===
    "claude-opus-4.6": LLMConfig(
        model="claude-opus-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        display_name="Claude Opus 4.6",
        provider="anthropic",
    ),
    "claude-sonnet-4.6": LLMConfig(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
    ),

    # === Local models via Ollama (free) ===
    "llama3-local": LLMConfig(
        model="llama3.1:8b",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        display_name="Llama3.1-8B",
        provider="ollama",
    ),
    "qwen2.5-local": LLMConfig(
        model="qwen2.5:7b",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        display_name="Qwen2.5-7B",
        provider="ollama",
    ),
}


def get_available_models() -> list[str]:
    """Return model keys that have API keys configured or are local."""
    available = []
    for key, cfg in MODELS.items():
        if cfg.provider == "ollama":
            available.append(key)
        elif cfg.provider == "anthropic":
            if cfg.api_key and HAS_ANTHROPIC:
                available.append(key)
        elif cfg.api_key:
            available.append(key)
    return available


def _query_openai(prompt: str, config: LLMConfig) -> str:
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return response.choices[0].message.content.strip()


def _query_anthropic(prompt: str, config: LLMConfig) -> str:
    if not HAS_ANTHROPIC:
        raise ImportError("pip install anthropic")
    client = anthropic.Anthropic(api_key=config.api_key)
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def query_llm(prompt: str, config: LLMConfig, retries: int = 3) -> str:
    """Send a prompt to an LLM and return the response text."""
    query_fn = _query_anthropic if config.provider == "anthropic" else _query_openai

    for attempt in range(retries):
        try:
            return query_fn(prompt, config)
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} after error: {e}, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed after {retries} attempts: {e}")
                return f"ERROR: {e}"

    return "ERROR: unreachable"


def parse_diagnosis(response: str, valid_labels: list[str]) -> str:
    """Extract the diagnosis label from LLM response."""
    response_clean = response.strip()

    for prefix in ["Diagnosis:", "Health stage:", "Condition:"]:
        if prefix in response_clean:
            after = response_clean.split(prefix)[-1].strip()
            candidate = after.split("\n")[0].strip().strip(".")
            for label in valid_labels:
                if label.lower() == candidate.lower():
                    return label

    for label in valid_labels:
        if label.lower() == response_clean.lower():
            return label

    for label in valid_labels:
        if label.lower() in response_clean.lower():
            return label

    return "UNPARSEABLE"
