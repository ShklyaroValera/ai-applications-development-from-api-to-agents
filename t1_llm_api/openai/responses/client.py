from openai import OpenAI, AsyncOpenAI

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class OpenAIResponsesClient(BaseOpenAIClient):
    """
    Client for OpenAI Responses API using the official SDK.

    This implementation uses the official OpenAI Python library to interact
    with the Responses API, which uses 'instructions' instead of system messages
    and 'input' instead of messages.

    Attributes:
        _client (OpenAI): Synchronous OpenAI client instance.
        _async_client (AsyncOpenAI): Asynchronous OpenAI client instance.
        Inherits all other attributes from BaseOpenAIClient.
    """

    def __init__(self, endpoint: str, model_name: str, system_prompt: str, api_key: str):
        """
        Initialize the OpenAI Responses client with SDK.

        Args:
            endpoint (str): The OpenAI API endpoint (for compatibility, not used by SDK).
            model_name (str): The OpenAI model to use (e.g., 'gpt-5').
            system_prompt (str): The instruction to guide the model's behavior.
            api_key (str): The OpenAI API key for authentication.
        """
        super().__init__(endpoint, model_name, system_prompt, api_key)
        # SDK adds the `Bearer ` prefix itself, so it gets the raw key (self._api_key is already Bearer-formatted)
        self._client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from OpenAI's Responses API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Note:
            Uses the Responses API format with 'instructions' and 'input' parameters.
            The response is printed to stdout before being returned.
        """
        response = self._client.responses.create(
            model=self._model_name,
            instructions=self._system_prompt,
            input=[message.to_dict() for message in messages],
        )
        content = response.output_text
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from OpenAI's Responses API.

        The response is streamed using event-based streaming, with each delta
        printed immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses the Responses API streaming format with event types.
            Listens for 'response.output_text.delta' events to build the response.
        """
        contents = []
        stream = await self._async_client.responses.create(
            model=self._model_name,
            instructions=self._system_prompt,
            input=[message.to_dict() for message in messages],
            stream=True,
        )
        async for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
                contents.append(event.delta)

        print()
        return Message(role=Role.ASSISTANT, content="".join(contents))