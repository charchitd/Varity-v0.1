import asyncio
import json
import httpx
from unittest.mock import AsyncMock, MagicMock
from varity.providers.gemini import GeminiProvider

async def test_gemini_retry_delay():
    # 1. Setup mock response with retryDelay
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.headers = httpx.Headers({})
    mock_resp.json.return_value = {
        "error": {
            "code": 429,
            "message": "Quota exceeded",
            "details": [
                {"retryDelay": "2s"}
            ]
        }
    }
    
    # Second response will be success
    success_resp = MagicMock(spec=httpx.Response)
    success_resp.status_code = 200
    success_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Success!"}]}}]
    }

    # 2. Setup provider
    provider = GeminiProvider(api_key="fake")
    provider._client.post = AsyncMock(side_effect=[mock_resp, success_resp])

    # 3. Track time
    import time
    start = time.time()
    result = await provider.complete("hello")
    end = time.time()
    
    print(f"Result: {result}")
    print(f"Duration: {end - start:.2f}s")
    assert result == "Success!"
    assert end - start >= 2.0

if __name__ == "__main__":
    asyncio.run(test_gemini_retry_delay())
