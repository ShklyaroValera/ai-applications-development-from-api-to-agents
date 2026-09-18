import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIResponsesClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Responses API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with the Responses API directly
    and handle its unique event-based streaming format.
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
            ValueError: If the API response contains no output text.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Uses the Responses API format with 'instructions' and 'input' parameters.
            The response is printed to stdout before being returned.
        """
        request_data = {
            "model": self._model_name,
            "instructions": self._system_prompt,
            "input": [message.to_dict() for message in messages],
        }

        response = requests.post(url=self._endpoint, headers=self._headers(), json=request_data)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        content = self._extract_output_text(response.json())
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with event-based streaming.

        The Responses API uses a different SSE format than Chat Completions,
        with explicit event types and data fields.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses event-based Server-Sent Events (SSE) format.
            Listens for 'response.output_text.delta' events to build the response.
            Each line with "event: " specifies the event type, followed by "data: " with the payload.
        """
        request_data = {
            "model": self._model_name,
            "instructions": self._system_prompt,
            "input": [message.to_dict() for message in messages],
            "stream": True,
        }
        contents = []

        async with aiohttp.ClientSession() as session:
            async with session.post(url=self._endpoint, headers=self._headers(), json=request_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                # SSE: `event: <type>` line, then `data: {json}` line, then an empty line
                event_type = None
                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        event_type = None
                    elif line_str.startswith("event: "):
                        event_type = line_str[len("event: "):].strip()
                    elif line_str.startswith("data: "):
                        data = json.loads(line_str[len("data: "):])
                        # `type` is duplicated inside the payload, use it as a fallback
                        current_type = event_type or data.get("type")
                        if current_type == "response.output_text.delta":
                            if delta := data.get("delta", ""):
                                print(delta, end="", flush=True)
                                contents.append(delta)
                        elif current_type == "response.completed":
                            break

        print()
        return Message(role=Role.ASSISTANT, content="".join(contents))

    def _headers(self) -> dict[str, str]:
        # self._api_key is already formatted as `Bearer <key>` by BaseOpenAIClient
        return {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_output_text(data: dict) -> str:
        texts = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
        if not texts:
            raise ValueError("No output text present in the response")
        return "".join(texts)
