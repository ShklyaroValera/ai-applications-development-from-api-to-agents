import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Google Gemini API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Gemini's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no candidates.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The URL is constructed by appending ':generateContent' to the model endpoint.
            Uses 'x-goog-api-key' header for authentication.
            Response candidates contain content parts that are concatenated.
        """
        url = f"{self._endpoint}/{self._model_name}:generateContent"

        response = requests.post(url=url, headers=self._headers(), json=self._request_data(messages, **kwargs))
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        candidates = response.json().get("candidates", [])
        if not candidates:
            raise ValueError("No candidates present in the response")

        content = self._extract_text(candidates[0])
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Gemini's SSE format, with text chunks
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The URL is constructed with ':streamGenerateContent?alt=sse' endpoint.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each SSE chunk contains candidates with content parts.
            Each text chunk is printed to stdout as it arrives.
        """
        url = f"{self._endpoint}/{self._model_name}:streamGenerateContent?alt=sse"
        contents = []

        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, headers=self._headers(), json=self._request_data(messages, **kwargs)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                # SSE: every chunk is `data: {json}` with the same structure as the non-streaming response
                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: "):
                        continue
                    data = json.loads(line_str[len("data: "):])
                    for candidate in data.get("candidates", [])[:1]:
                        if text := self._extract_text(candidate):
                            print(text, end="", flush=True)
                            contents.append(text)

        print()
        return Message(role=Role.ASSISTANT, content="".join(contents))

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

    def _request_data(self, messages: list[Message], **kwargs) -> dict:
        return {
            "system_instruction": {"parts": [{"text": self._system_prompt}]},
            "contents": [
                {
                    # Gemini uses role `model` instead of `assistant`
                    "role": "model" if message.role == Role.ASSISTANT else "user",
                    "parts": [{"text": message.content}],
                }
                for message in messages
            ],
            "generationConfig": {"maxOutputTokens": kwargs.get("max_tokens", 1024)},
        }

    @staticmethod
    def _extract_text(candidate: dict) -> str:
        parts = candidate.get("content", {}).get("parts", [])
        # skip thought-summary parts, keep only answer text
        return "".join(part.get("text", "") for part in parts if not part.get("thought"))