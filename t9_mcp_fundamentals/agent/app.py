import os
import sys
import asyncio
import json
from pathlib import Path

from mcp import Resource
from mcp.types import Prompt

from commons.constants import OPENAI_API_KEY
from commons.models.message import Message
from commons.models.role import Role
from t9_mcp_fundamentals.agent.agent import AgentMCPFundamentals
from t9_mcp_fundamentals.agent.mcp_clients.base import MCPClient
from t9_mcp_fundamentals.agent.mcp_clients.http import HttpMCPClient
from t9_mcp_fundamentals.agent.mcp_clients.multi import MultiMCPClient
from t9_mcp_fundamentals.agent.mcp_clients.stdio import StdioMCPClient
from t9_mcp_fundamentals.agent.prompts import SYSTEM_PROMPT


PROJECT_ROOT = Path(__file__).parent.parent.parent  # .../ai-applications-development-from-api-to-agents
STDIO_SERVER_PATH = PROJECT_ROOT / "t9_mcp_fundamentals" / "mcp_server" / "stdio_server.py"

USERS_MCP_URL = "http://localhost:8005/mcp"
# README fetch server https://remote.mcpservers.org/fetch/mcp no longer resolves (NXDOMAIN),
# GitMCP exposes an equivalent remote tool `fetch_generic_url_content`. Override with env var FETCH_MCP_URL.
FETCH_MCP_URL = os.getenv("FETCH_MCP_URL", "https://gitmcp.io/docs")  # remote server: tools only, no resources/prompts

# Switch between README variants with env var `MCP_MODE` (default is the first README variant - `http`):
#   http   -> HttpMCPClient to users-management MCP server (http_server.py must be running on :8005)
#   fetch  -> HttpMCPClient to remote `fetch` MCP server
#   stdio  -> StdioMCPClient that spawns local stdio_server.py
#   docker -> StdioMCPClient that runs `mcp/duckduckgo:latest` docker image
#   multi  -> OPTIONAL: users-management (http) + fetch MCP servers at once
# `MCP_SERVER_URL` overrides the URL in `http` mode.
MCP_MODE = os.getenv("MCP_MODE", "http").lower()


def create_mcp_client() -> MCPClient:
    if MCP_MODE == "http":
        return HttpMCPClient(mcp_server_url=os.getenv("MCP_SERVER_URL", USERS_MCP_URL))
    if MCP_MODE == "fetch":
        return HttpMCPClient(mcp_server_url=FETCH_MCP_URL)
    if MCP_MODE == "stdio":
        return StdioMCPClient(
            command=sys.executable,  # use the same venv Python, not bare "python"
            args=[str(STDIO_SERVER_PATH)],
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},  # inherit env + add project root
        )
    if MCP_MODE == "docker":
        return StdioMCPClient(docker_image="mcp/duckduckgo:latest")
    if MCP_MODE == "multi":
        return MultiMCPClient(
            clients=[
                HttpMCPClient(mcp_server_url=USERS_MCP_URL),
                HttpMCPClient(mcp_server_url=FETCH_MCP_URL),
            ]
        )
    raise ValueError(f"Unknown MCP_MODE `{MCP_MODE}`. Use one of: http, fetch, stdio, docker, multi")


async def main():
    async with create_mcp_client() as mcp_client:
        print("\n=== Available Resources ===")
        resources: list[Resource] = await mcp_client.get_resources()
        for resource in resources:
            print(resource)

        print("\n=== Available Tools ===")
        tools: list[dict] = await mcp_client.get_tools()
        for tool in tools:
            print(json.dumps(tool, indent=2))

        agent = AgentMCPFundamentals(
            api_key=OPENAI_API_KEY,
            model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
            tools=tools,
            mcp_client=mcp_client,
        )

        messages: list[Message] = [Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)]

        print("\n=== Available Prompts ===")
        prompts: list[Prompt] = await mcp_client.get_prompts()
        for prompt in prompts:
            print(prompt)
            content = await mcp_client.get_prompt(prompt.name)
            print(content)
            messages.append(
                Message(
                    role=Role.USER,
                    content=f"## Prompt provided by MCP server:\n{prompt.description}\n{content}",
                )
            )

        print("MCP-based Agent is ready! Type your query or 'exit' to exit.")
        while True:
            try:
                user_input = input("\n> ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break

            messages.append(Message(role=Role.USER, content=user_input))
            ai_message: Message = await agent.get_response(messages)
            messages.append(ai_message)


if __name__ == "__main__":
    asyncio.run(main())
