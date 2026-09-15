import json
import logging
from collections import defaultdict
from typing import AsyncGenerator, Any

from openai import AsyncOpenAI

from t13_final_task.task.agent.models import Message
from t13_final_task.task.agent.models import Role
from t13_final_task.task.agent.guardrail import UMSDataGuardrail
from t13_final_task.task.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class UMSAgent:
    """Handles AI model interactions and integrates with MCP client"""

    def __init__(
            self,
            api_key: str,
            model: str,
            tools: list[BaseTool]
    ):
        self.tools: dict[str, BaseTool] = {tool.name: tool for tool in tools} if tools else {}
        self._tools_schemas: list[dict[str, Any]] = [tool.schema for tool in tools] if tools else []
        self.model = model
        self.async_openai = AsyncOpenAI(api_key=api_key)
        self.guardrail = UMSDataGuardrail()

    def _request_data(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        request_data: dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in messages],
            "stream": stream,
        }
        if self._tools_schemas:
            request_data["tools"] = self._tools_schemas
        return request_data

    async def response(self, messages: list[Message]) -> Message:
        """Non-streaming completion with tool calling support"""
        logger.debug(
            "Creating non-streaming completion",
            extra={"message_count": len(messages), "model": self.model}
        )

        response = await self.async_openai.chat.completions.create(**self._request_data(messages, stream=False))
        choice_message = response.choices[0].message
        ai_message = Message(role=Role.ASSISTANT, content=choice_message.content or "")
        if choice_message.tool_calls:
            ai_message.tool_calls = [tool_call.model_dump() for tool_call in choice_message.tool_calls]

        if ai_message.tool_calls:
            messages.append(ai_message)
            await self._call_tools(ai_message, messages)
            return await self.response(messages)

        messages.append(ai_message)
        return ai_message

    async def stream_response(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """
        Streaming completion with tool calling support.
        Yields SSE-formatted chunks.
        """
        logger.debug(
            "Creating streaming completion",
            extra={"message_count": len(messages), "model": self.model}
        )

        stream = await self.async_openai.chat.completions.create(**self._request_data(messages, stream=True))

        content_buffer = ""
        tool_deltas = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_buffer += delta.content
                yield _sse({"choices": [{"delta": {"content": delta.content}, "index": 0, "finish_reason": None}]})
            if delta.tool_calls:
                tool_deltas.extend(delta.tool_calls)

        if tool_deltas:
            tool_calls = self._collect_tool_calls(tool_deltas)
            ai_message = Message(role=Role.ASSISTANT, content=content_buffer, tool_calls=tool_calls)
            messages.append(ai_message)

            for tool_call in tool_calls:
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    tool_args = {}
                yield _sse({"tool_activity": {
                    "type": "call",
                    "name": tool_call["function"]["name"],
                    "arguments": tool_args,
                }})

            prev_len = len(messages)
            await self._call_tools(ai_message, messages)
            for msg in messages[prev_len:]:
                yield _sse({"tool_activity": {"type": "result", "name": msg.name, "content": msg.content}})

            async for chunk in self.stream_response(messages):
                yield chunk
            return

        messages.append(Message(role=Role.ASSISTANT, content=content_buffer))
        yield _sse({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]})
        yield "data: [DONE]\n\n"

    def _collect_tool_calls(self, tool_deltas) -> list[dict[str, Any]]:
        """Convert streaming tool call deltas to complete tool calls"""
        tool_dict: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"id": None, "function": {"arguments": "", "name": None}, "type": None}
        )
        for delta in tool_deltas:
            tool_call = tool_dict[delta.index]
            if delta.id:
                tool_call["id"] = delta.id
            if delta.function:
                if delta.function.name:
                    tool_call["function"]["name"] = delta.function.name
                if delta.function.arguments:
                    tool_call["function"]["arguments"] += delta.function.arguments
            if delta.type:
                tool_call["type"] = delta.type

        tool_calls = list(tool_dict.values())
        for tool_call in tool_calls:
            tool_call["type"] = tool_call["type"] or "function"
        return tool_calls

    async def _call_tools(self, ai_message: Message, messages: list[Message], silent: bool = False):
        """Execute tool calls using MCP client"""
        for tool_call in ai_message.tool_calls or []:
            tool_call_id = tool_call["id"]
            tool_name = tool_call["function"]["name"]
            try:
                arguments = json.loads(tool_call["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                messages.append(Message(
                    role=Role.TOOL,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    content=f"Error: invalid JSON arguments for tool '{tool_name}': {e}",
                ))
                continue

            tool = self.tools.get(tool_name)
            if tool is None:
                messages.append(Message(
                    role=Role.TOOL,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    content=f"Error: tool '{tool_name}' not found",
                ))
                continue

            if not silent:
                logger.info("Calling tool", extra={"tool_name": tool_name, "arguments": arguments})

            result = await tool.execute(tool_call_id, arguments)
            result.name = tool_name
            # Input guardrail: never send credit card / salary data to the model or store it in history
            result.content = self.guardrail.redact(str(result.content))
            messages.append(result)
