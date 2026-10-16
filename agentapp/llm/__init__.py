"""LLM provider adapters behind one provider-agnostic interface."""

from __future__ import annotations

from ..config import Settings
from .base import LLMError, LLMProvider

SUPPORTED = ("anthropic", "ollama")


def make_provider(settings: Settings) -> LLMProvider:
    """Build the configured provider. Imports are lazy so unused SDKs are not required."""
    if settings.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            settings.model,
            max_tokens=settings.max_tokens,
            effort=settings.effort,
            fallbacks=settings.fallbacks,
        )
    if settings.provider == "ollama":
        from .ollama_provider import OllamaProvider

        if settings.model.startswith("claude-"):
            raise LLMError(
                "AGENT_PROVIDER=ollama needs AGENT_MODEL set to a local model, "
                "e.g. llama3.1:8b-instruct-q4_K_M (see `ollama list`)"
            )
        return OllamaProvider(
            settings.model,
            host=settings.ollama_host,
            num_ctx=settings.ollama_num_ctx,
            max_tokens=settings.ollama_num_predict,
            timeout_s=settings.ollama_timeout_s,
        )
    raise LLMError(
        f"unknown provider {settings.provider!r}; supported: {', '.join(SUPPORTED)}. "
        "Add an adapter in agentapp/llm/ implementing LLMProvider.complete()."
    )
