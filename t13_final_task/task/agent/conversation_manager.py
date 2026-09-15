import json
import logging
import uuid
from datetime import datetime, UTC
from typing import Optional, AsyncGenerator

import redis.asyncio as redis

from t13_final_task.task.agent.models import Message
from t13_final_task.task.agent.models import Role
from t13_final_task.task.agent.ums_agent import UMSAgent

logger = logging.getLogger(__name__)

_CONVERSATION_PREFIX = "conversation:"
_CONVERSATION_LIST_KEY = "conversations:list"

class ConversationManager:
    """Manages conversation lifecycle including AI interactions and persistence"""

    def __init__(self, ums_agent: UMSAgent, redis_client: redis.Redis, system_prompt: str):
        self.ums_agent = ums_agent
        self.redis = redis_client
        self._system_prompt = system_prompt
        logger.info("ConversationManager initialized")

    async def create_conversation(self, title: str) -> dict:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        conversation = {
            "id": conversation_id,
            "title": title,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        await self.redis.set(_CONVERSATION_PREFIX + conversation_id, json.dumps(conversation))
        await self.redis.zadd(_CONVERSATION_LIST_KEY, {conversation_id: datetime.now(UTC).timestamp()})
        logger.info("Conversation created", extra={"conversation_id": conversation_id})
        return conversation

    async def list_conversations(self) -> list[dict]:
        """List all conversations sorted by last update time"""
        conversation_ids = await self.redis.zrevrange(_CONVERSATION_LIST_KEY, 0, -1)
        conversations = []
        for conv_id in conversation_ids:
            if isinstance(conv_id, bytes):
                conv_id = conv_id.decode("utf-8")
            data = await self.redis.get(_CONVERSATION_PREFIX + conv_id)
            if not data:
                continue
            conversation = json.loads(data)
            conversations.append({
                "id": conversation["id"],
                "title": conversation["title"],
                "created_at": conversation["created_at"],
                "updated_at": conversation["updated_at"],
                "message_count": len(conversation["messages"]),
            })
        return conversations

    async def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a specific conversation"""
        data = await self.redis.get(_CONVERSATION_PREFIX + conversation_id)
        if not data:
            return None
        return json.loads(data)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        deleted = await self.redis.delete(_CONVERSATION_PREFIX + conversation_id)
        if deleted == 0:
            return False
        await self.redis.zrem(_CONVERSATION_LIST_KEY, conversation_id)
        logger.info("Conversation deleted", extra={"conversation_id": conversation_id})
        return True

    async def chat(
            self,
            user_message: Message,
            conversation_id: str,
            stream: bool = False
    ):
        """
        Process chat messages and return AI response.
        Automatically saves conversation state.
        """
        logger.info(
            "Processing chat request",
            extra={
                "conversation_id": conversation_id,
                "stream": stream,
                "message_role": user_message.role
            }
        )

        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation with id {conversation_id} not found")

        messages = [Message(**msg) for msg in conversation["messages"]]
        if not messages:
            messages.append(Message(role=Role.SYSTEM, content=self._system_prompt))
        messages.append(user_message)

        if stream:
            return self._stream_chat(conversation_id, messages)
        return await self._non_stream_chat(conversation_id, messages)

    async def _stream_chat(
            self,
            conversation_id: str,
            messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        """Handle streaming chat with automatic saving"""
        logger.debug("Starting streaming chat", extra={"conversation_id": conversation_id})

        yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
        try:
            async for chunk in self.ums_agent.stream_response(messages):
                yield chunk
        finally:
            await self._save_conversation_messages(conversation_id, messages)

    async def _non_stream_chat(
            self,
            conversation_id: str,
            messages: list[Message],
    ) -> dict:
        """Handle non-streaming chat"""
        logger.debug("Starting non-streaming chat", extra={"conversation_id": conversation_id})

        ai_message = await self.ums_agent.response(messages)
        await self._save_conversation_messages(conversation_id, messages)
        return {"content": ai_message.content or "", "conversation_id": conversation_id}

    async def _save_conversation_messages(
            self,
            conversation_id: str,
            messages: list[Message]
    ):
        """Save or update conversation messages"""
        logger.debug(
            "Saving conversation messages",
            extra={"conversation_id": conversation_id}
        )

        data = await self.redis.get(_CONVERSATION_PREFIX + conversation_id)
        if not data:
            logger.warning("Conversation disappeared before save", extra={"conversation_id": conversation_id})
            return
        conversation = json.loads(data)
        conversation["messages"] = [msg.to_dict() for msg in messages]
        conversation["updated_at"] = datetime.now(UTC).isoformat()
        await self._save_conversation(conversation)

    async def _save_conversation(self, conversation: dict):
        """Internal method to persist conversation to Redis"""
        conversation_id = conversation["id"]
        await self.redis.set(_CONVERSATION_PREFIX + conversation_id, json.dumps(conversation))
        await self.redis.zadd(_CONVERSATION_LIST_KEY, {conversation_id: datetime.now(UTC).timestamp()})
