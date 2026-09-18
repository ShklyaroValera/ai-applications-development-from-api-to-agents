import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Chat Completions API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, providing more control over the HTTP layer and demonstrating
    how to interact with the API directly.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no choices.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        request_data = {
            "model": self._model_name,
            "messages": self._prepare_messages(messages),
        }

        response = requests.post(url=self._endpoint, headers=self._headers(), json=request_data)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        choices = response.json().get("choices", [])
        if not choices:
            raise ValueError("No choices present in the response")

        content = choices[0].get("message", {}).get("content") or ""
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed token-by-token using OpenAI's SSE format,
        with each chunk printed immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
        """
        request_data = {
            "model": self._model_name,
            "messages": self._prepare_messages(messages),
            "stream": True,
        }
        contents = []

        async with aiohttp.ClientSession() as session:
            async with session.post(url=self._endpoint, headers=self._headers(), json=request_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                # SSE: every event is a line `data: {json}`, the stream ends with `data: [DONE]`
                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: "):
                        continue
                    data = line_str[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    if snippet := self._get_content_snippet(data):
                        print(snippet, end="", flush=True)
                        contents.append(snippet)

        print()
        return Message(role=Role.ASSISTANT, content="".join(contents))

    def _headers(self) -> dict[str, str]:
        # self._api_key is already formatted as `Bearer <key>` by BaseOpenAIClient
        return {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        return [
            {"role": Role.SYSTEM.value, "content": self._system_prompt},
            *[message.to_dict() for message in messages],
        ]

    @staticmethod
    def _get_content_snippet(data: str) -> str:
        chunk = json.loads(data)
        if choices := chunk.get("choices"):
            return choices[0].get("delta", {}).get("content") or ""
        return ""
