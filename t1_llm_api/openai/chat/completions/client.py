from openai import OpenAI, AsyncOpenAI

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class OpenAIClient(BaseOpenAIClient):
    """
    Client for OpenAI Chat Completions API using the official SDK.

    This implementation uses the official OpenAI Python library to interact
    with the Chat Completions API, providing both synchronous and streaming
    response capabilities.

    Attributes:
        _client (OpenAI): Synchronous OpenAI client instance.
        _async_client (AsyncOpenAI): Asynchronous OpenAI client instance.
        Inherits all other attributes from BaseOpenAIClient.
    """

    def __init__(self, endpoint: str, model_name: str, system_prompt: str, api_key: str):
        """
        Initialize the OpenAI Chat Completions client with SDK.

        Args:
            endpoint (str): The OpenAI API endpoint (for compatibility, not used by SDK).
            model_name (str): The OpenAI model to use (e.g., 'gpt-5').
            system_prompt (str): The system message to guide the model's behavior.
            api_key (str): The OpenAI API key for authentication.
        """
        super().__init__(endpoint, model_name, system_prompt, api_key)
        # SDK adds the `Bearer ` prefix itself, so it gets the raw key (self._api_key is already Bearer-formatted)
        self._client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        return [
            {"role": Role.SYSTEM.value, "content": self._system_prompt},
            *[message.to_dict() for message in messages],
        ]

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from OpenAI's Chat Completions API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=self._prepare_messages(messages),
        )
        content = response.choices[0].message.content or ""
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from OpenAI's Chat Completions API.

        The response is streamed token-by-token, with each chunk printed
        immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives for real-time display.
        """
        contents = []
        stream = await self._async_client.chat.completions.create(
            model=self._model_name,
            messages=self._prepare_messages(messages),
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            if delta := chunk.choices[0].delta.content:
                print(delta, end="", flush=True)
                contents.append(delta)

        print()
        return Message(role=Role.ASSISTANT, content="".join(contents))
