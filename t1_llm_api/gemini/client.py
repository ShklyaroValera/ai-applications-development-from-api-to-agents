from google import genai
from google.genai import types

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class GeminiAIClient(AIClient):
    """
    Client for Google Gemini API using the official SDK.

    This implementation uses the official Google GenAI Python library to interact
    with Gemini models, providing both synchronous and streaming response capabilities.

    Attributes:
        _client (genai.Client): Google GenAI client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Gemini client with SDK.

        Args:
            endpoint (str): The Gemini API endpoint (for compatibility, not used by SDK).
            model_name (str): The Gemini model to use (e.g., 'gemini-3-flash-preview').
            api_key (str): The Google API key for authentication.
            system_prompt (str): The system instruction to guide the model's behavior.
        """
        super().__init__(endpoint, model_name, api_key, system_prompt)
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _to_gemini_contents(messages: list[Message]) -> list[types.Content]:
        # Gemini uses role `model` instead of `assistant`
        return [
            types.Content(
                role="model" if message.role == Role.ASSISTANT else "user",
                parts=[types.Part(text=message.content)],
            )
            for message in messages
        ]

    def _config(self, **kwargs) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            max_output_tokens=kwargs.get("max_tokens", 1024),
        )

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from Google's Gemini API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Note:
            Gemini uses 'system_instruction' parameter for system-level guidance.
            The response is printed to stdout before being returned.
        """
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=self._to_gemini_contents(messages),
            config=self._config(**kwargs),
        )
        content = response.text or ""
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from Google's Gemini API.

        The response is streamed chunk-by-chunk, with each text chunk printed
        immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            Uses the async streaming interface provided by the Gemini SDK.
            Each chunk's text is printed to stdout as it arrives.
        """
        contents = []
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model_name,
            contents=self._to_gemini_contents(messages),
            config=self._config(**kwargs),
        )
        async for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                contents.append(chunk.text)

        print()
        return Message(role=Role.ASSISTANT, content="".join(contents))