"""Unit tests for the Varity provider system.

All tests use MockProvider or mocked httpx responses — no real API calls.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from varity.exceptions import ConfigError, DecompositionError, ProviderError
from varity.providers import get_provider
from varity.providers.anthropic import AnthropicProvider
from varity.providers.base import BaseLLMProvider
from varity.providers.gemini import GeminiProvider
from varity.providers.openai import OpenAIProvider


# ---------------------------------------------------------------------------
# MockProvider — a minimal concrete provider for unit testing
# ---------------------------------------------------------------------------

class MockProvider(BaseLLMProvider):
    """Concrete provider that returns a pre-set response without any I/O."""

    def __init__(self, response: str = '{"ok": true}', **kwargs: Any) -> None:
        super().__init__(api_key="mock-key", model="mock-model", **kwargs)
        self._response = response

    async def complete(self, prompt: str, system: str = "") -> str:
        """Return the pre-set response string."""
        return self._response


# ---------------------------------------------------------------------------
# Helper to build a fake httpx.Response
# ---------------------------------------------------------------------------

def _fake_response(status_code: int, body: dict[str, Any]) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body)
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


# ===========================================================================
# Test 1 — Provider initialises correctly with key and model
# ===========================================================================

def test_provider_initialises_with_key_and_model() -> None:
    """Provider stores model name; api_key is not publicly accessible."""
    provider = AnthropicProvider(api_key="sk-ant-test", model="claude-3-haiku-20240307")
    assert provider.model == "claude-3-haiku-20240307"
    # API key must NOT appear on public attribute
    assert not hasattr(provider, "api_key")


# ===========================================================================
# Test 2 — Factory returns correct provider class
# ===========================================================================

@pytest.mark.parametrize(
    "name, expected_cls",
    [
        ("anthropic", AnthropicProvider),
        ("openai", OpenAIProvider),
        ("gemini", GeminiProvider),
    ],
)
def test_factory_returns_correct_class(
    name: str, expected_cls: type[BaseLLMProvider]
) -> None:
    """get_provider() returns the correct concrete provider class."""
    provider = get_provider(name, api_key="dummy-key")
    assert isinstance(provider, expected_cls)


# ===========================================================================
# Test 3 — Invalid provider name raises ConfigError
# ===========================================================================

def test_invalid_provider_name_raises_config_error() -> None:
    """get_provider() raises ConfigError for unknown provider names."""
    with pytest.raises(ConfigError, match="Unknown provider"):
        get_provider("cohere", api_key="dummy-key")


# ===========================================================================
# Test 4 — complete_json parses valid JSON
# ===========================================================================

@pytest.mark.asyncio
async def test_complete_json_parses_valid_json() -> None:
    """complete_json() returns a dict when the provider emits valid JSON."""
    provider = MockProvider(response='{"verdict": "supported", "confidence": 0.9}')
    result = await provider.complete_json("some prompt")
    assert result == {"verdict": "supported", "confidence": 0.9}


# ===========================================================================
# Test 5 — complete_json raises DecompositionError on invalid JSON
# ===========================================================================

@pytest.mark.asyncio
async def test_complete_json_raises_on_invalid_json() -> None:
    """complete_json() raises DecompositionError when response is not JSON."""
    provider = MockProvider(response="This is just plain text, not JSON.")
    with pytest.raises(DecompositionError, match="not valid JSON"):
        await provider.complete_json("some prompt")


# ===========================================================================
# Test 6 — Retry logic triggers on 429
# ===========================================================================

@pytest.mark.asyncio
async def test_retry_triggers_on_429() -> None:
    """_with_retry retries on HTTP 429 and eventually raises ProviderError."""
    call_count = 0

    async def _flaky_post() -> httpx.Response:
        nonlocal call_count
        call_count += 1
        resp = _fake_response(429, {"error": "rate limited"})
        raise httpx.HTTPStatusError("429", request=MagicMock(), response=resp)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.HTTPStatusError):
            await BaseLLMProvider._with_retry(_flaky_post)

    # Should have tried 3 times (len(_RETRY_DELAYS) == 3)
    assert call_count == 3


# ===========================================================================
# Test 7 — 401 raises ProviderError with "Invalid API key"
# ===========================================================================

@pytest.mark.asyncio
async def test_401_raises_provider_error_with_invalid_key_message() -> None:
    """A 401 response raises ProviderError mentioning 'Invalid API key'."""
    provider = AnthropicProvider(api_key="bad-key")

    fake_resp = _fake_response(401, {"error": "unauthorized"})
    fake_resp.raise_for_status = MagicMock()  # 401 is handled before raise_for_status

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = fake_resp
        with pytest.raises(ProviderError, match="Invalid API key"):
            await provider.complete("Hello")


# ===========================================================================
# Test 8 — Context manager opens and closes the client
# ===========================================================================

@pytest.mark.asyncio
async def test_context_manager_opens_and_closes_client() -> None:
    """async with provider: closes the httpx client on exit."""
    provider = MockProvider()
    close_called = False
    original_close = provider.close

    async def _patched_close() -> None:
        nonlocal close_called
        close_called = True
        await original_close()

    provider.close = _patched_close  # type: ignore[method-assign]

    async with provider:
        pass

    assert close_called, "close() must be called on __aexit__"
