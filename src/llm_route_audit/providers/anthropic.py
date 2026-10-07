"""Replay requests on Claude models through the official Anthropic SDK.

Server-side refusal fallbacks are deliberately not enabled: a fallback would answer with a
different model, and the replay must measure the candidate itself. Refusals are recorded
with status "refusal" instead.
"""

from typing import Any

import anthropic

from llm_route_audit.candidates import Candidate
from llm_route_audit.providers.base import Completion, ProviderError
from llm_route_audit.records import Message

MAX_RETRIES = 5
STATUS_BY_STOP_REASON = {"refusal": "refusal", "max_tokens": "truncated"}


def build_request(candidate: Candidate, messages: list[Message]) -> dict[str, Any]:
    """Messages API arguments. Logged system messages are joined into the `system` field."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    request: dict[str, Any] = {
        "model": candidate.api_model,
        "max_tokens": candidate.max_tokens,
        "messages": [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ],
    }
    if system:
        request["system"] = system
    if candidate.effort:
        request["output_config"] = {"effort": candidate.effort}
    return request


def parse_response(response: Any) -> Completion:
    usage = response.usage
    return Completion(
        text="".join(block.text for block in response.content if block.type == "text"),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
        status=STATUS_BY_STOP_REASON.get(response.stop_reason, "ok"),
    )


class AnthropicProvider:
    def __init__(self, client: Any = None) -> None:
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                self._client = anthropic.Anthropic(max_retries=MAX_RETRIES)
            except anthropic.AnthropicError as e:
                raise ProviderError(
                    f"Could not create the Anthropic client: {e}. Set ANTHROPIC_API_KEY.",
                    fatal=True,
                ) from e
        return self._client

    def complete(self, candidate: Candidate, messages: list[Message]) -> Completion:
        client = self._get_client()
        try:
            response = client.messages.create(**build_request(candidate, messages))
        except anthropic.AuthenticationError as e:
            raise ProviderError(
                "Anthropic rejected the API key. Check ANTHROPIC_API_KEY.", fatal=True
            ) from e
        except anthropic.PermissionDeniedError as e:
            raise ProviderError(f"Permission denied: {e.message}", fatal=True) from e
        except (anthropic.BadRequestError, anthropic.NotFoundError) as e:
            raise ProviderError(f"{candidate.label}: {e.message}", disable=True) from e
        except anthropic.RateLimitError as e:
            raise ProviderError(f"Rate limited after {MAX_RETRIES} retries") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("Could not reach the Anthropic API (network error)") from e
        return parse_response(response)
