import logging
import os
import sys
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from t13_final_task.task.agent.clients.http_mcp_client import HttpMcpClient
from t13_final_task.task.agent.clients.stdio_mcp_client import StdioMcpClient
from t13_final_task.task.agent.conversation_manager import ConversationManager
from t13_final_task.task.agent.tools.base import BaseTool
from t13_final_task.task.agent.tools.mcp_tool import McpTool
from t13_final_task.task.agent.tools.read_skill_tool import ReadSkillTool
from t13_final_task.task.agent.ums_agent import UMSAgent
from t13_final_task.task.agent.models import SkillMetadata, load_skills, Message

SKILLS_DIR = Path(__file__).parent.parent / "_skills"


def _build_available_skills_xml(skills: list[SkillMetadata]) -> str:
    root = ET.Element("available_skills")
    for skill in skills:
        el = ET.SubElement(root, "skill", name=skill.name)
        ET.SubElement(el, "description").text = skill.description
        if skill.license:
            ET.SubElement(el, "license").text = str(skill.license)
        if skill.compatibility:
            ET.SubElement(el, "compatibility").text = skill.compatibility
        if skill.metadata:
            meta = ET.SubElement(el, "metadata")
            for key, value in skill.metadata.items():
                ET.SubElement(meta, str(key)).text = str(value)
        if skill.allowed_tools:
            ET.SubElement(el, "allowed-tools").text = " ".join(skill.allowed_tools)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def build_system_prompt(skills: list[SkillMetadata]) -> str:
    return f"""\
You are the Users Management Agent, an AI assistant with access to agent skills and tools.
You help users search, create, update and delete users in the Users Management Service (UMS)
and can search the web to enrich user profiles or answer general questions.

{_build_available_skills_xml(skills)}

## How to use skills

1. When the user's request matches a skill, call `read_skill` with path "/<skill-name>/SKILL.md"
   to load its full instructions, then follow them precisely.
2. If the instructions reference additional files, read them on demand with `read_skill`.
3. Always read the relevant SKILL.md before performing the task.

## Rules

- Never reveal credit card data or salary values; if a tool result contains "***", keep it masked.
- Always ask for explicit confirmation before adding or deleting a user; a plain "yes" to your question is sufficient confirmation.
- Never invent user data: use tool results and web search results only.
- Keep answers concise and use Markdown formatting.\
"""


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

conversation_manager: Optional[ConversationManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCP clients, Redis, and ConversationManager on startup"""
    global conversation_manager

    skills = load_skills(SKILLS_DIR)
    logger.info("Skills loaded", extra={"skills": [skill.name for skill in skills]})
    system_prompt = build_system_prompt(skills)
    logger.info(f"System prompt:\n{system_prompt}")

    tools: list[BaseTool] = [ReadSkillTool(skills_dir=SKILLS_DIR)]

    ums_mcp_url = os.getenv("UMS_MCP_URL", "http://localhost:8005/mcp")
    ums_mcp_client = await HttpMcpClient.create(ums_mcp_url)
    for mcp_tool_model in await ums_mcp_client.get_tools():
        tools.append(McpTool(client=ums_mcp_client, mcp_tool_model=mcp_tool_model))

    ddg_mcp_client = await StdioMcpClient.create(docker_image="khshanovskyi/ddg-mcp-server:latest")
    for mcp_tool_model in await ddg_mcp_client.get_tools():
        tools.append(McpTool(client=ddg_mcp_client, mcp_tool_model=mcp_tool_model))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    agent = UMSAgent(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        tools=tools,
    )

    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True,
    )
    await redis_client.ping()
    logger.info("Redis connection established")

    conversation_manager = ConversationManager(agent, redis_client, system_prompt=system_prompt)

    yield

    await redis_client.aclose()
    logger.info("Redis connection closed")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_manager() -> ConversationManager:
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return conversation_manager


# Request/Response Models
class ChatRequest(BaseModel):
    message: Message
    stream: bool = True


class ChatResponse(BaseModel):
    content: str
    conversation_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class CreateConversationRequest(BaseModel):
    title: str = None


# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "conversation_manager_initialized": conversation_manager is not None
    }


@app.post("/conversations")
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation"""
    manager = _require_manager()
    return await manager.create_conversation(request.title or "New conversation")


@app.get("/conversations")
async def list_conversations():
    """List all conversations sorted by last update time"""
    manager = _require_manager()
    conversations = await manager.list_conversations()
    return [ConversationSummary(**conv_dict) for conv_dict in conversations]


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    manager = _require_manager()
    conversation = await manager.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    manager = _require_manager()
    deleted = await manager.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": f"Conversation {conversation_id} deleted successfully"}


@app.post("/conversations/{conversation_id}/chat")
async def chat(conversation_id: str, request: ChatRequest):
    """
    Chat endpoint that processes messages and returns assistant response.
    Supports both streaming and non-streaming modes.
    Automatically saves conversation state.
    """
    manager = _require_manager()
    try:
        result = await manager.chat(
            user_message=request.message,
            conversation_id=conversation_id,
            stream=request.stream,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if request.stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return ChatResponse(**result)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8011,
        log_level="debug",
    )