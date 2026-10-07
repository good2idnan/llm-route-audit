"""Model providers that replay requests. Each one turns a message list into a Completion."""

from llm_route_audit.providers.base import Completion, Provider, ProviderError


def get_provider(name: str) -> Provider:
    """Create the provider client for a candidate's `provider` field."""
    if name == "anthropic":
        from llm_route_audit.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if name == "ollama":
        from llm_route_audit.providers.ollama import OllamaProvider

        return OllamaProvider()
    if name == "openrouter":
        from llm_route_audit.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider()
    raise ValueError(f"unknown provider '{name}'")


__all__ = ["Completion", "Provider", "ProviderError", "get_provider"]
