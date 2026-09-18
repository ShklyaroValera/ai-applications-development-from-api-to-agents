import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomAnthropicAIClient(AIClient):
    """
    Custom HTTP client for Anthropic's Claude API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Claude's API directly
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
            ValueError: If the API response contains no content blocks.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Requires 'x-api-key' header and 'anthropic-version' header.
            Claude's API returns content as an array of content blocks.
            The response is printed to stdout before being returned.
        """
        request_data = self._request_data(messages, **kwargs)

        response = requests.post(url=self._endpoint, headers=self._headers(), json=request_data)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        content_blocks = response.json().get("content", [])
        if not content_blocks:
            raise ValueError("No content blocks present in the response")

        content = "".join(block.get("text", "") for block in content_blocks if block.get("type") == "text")
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Anthropic's SSE format, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Listens for 'content_block_delta' events with 'text_delta' type.
            Stops processing when 'message_stop' event is received.
            Each delta is printed to stdout as it arrives.
        """
        request_data = self._request_data(messages, **kwargs)
        request_data["stream"] = True
        contents = []

        async with aiohttp.ClientSession() as session:
            async with session.post(url=self._endpoint, headers=self._headers(), json=request_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                # SSE: `event: <type>` + `data: {json}`; the event type is also present in the payload `type`
                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: "):
                        continue
                    data = json.loads(line_str[len("data: "):])
                    event_type = data.get("type")
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta" and (text := delta.get("text")):
                            print(text, end="", flush=True)
                            contents.append(text)
                    elif event_type == "error":
                        raise Exception(f"Stream error: {data.get('error')}")
                    elif event_type == "message_stop":
                        break

        print()
        return Message(role=Role.ASSISTANT, content="".join(contents))

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _request_data(self, messages: list[Message], **kwargs) -> dict:
        return {
            "model": self._model_name,
            "system": self._system_prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [message.to_dict() for message in messages],
        }

