import asyncio
import json
import os

from commons.constants import OPENAI_API_KEY, DEFAULT_SYSTEM_PROMPT
from commons.models.message import Message
from commons.models.role import Role
from t11_mcp_auth.agent._agent import AgentMCPAuth
from t11_mcp_auth.agent.mcp_clients.api_key_mcp_client import ApiKeyMCPClient
from t11_mcp_auth.agent.mcp_clients.oauth_mcp_client import OauthHttpMCPClient

MCP_API_KEY: str = "dev-secret-key"

# Which MCP server/auth strategy to use: "api_key" (Part 1, default) or "oauth" (Part 2).
# Switch via env var: MCP_AUTH_MODE=oauth
MCP_AUTH_MODE: str = os.getenv("MCP_AUTH_MODE", "api_key").lower()
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.2")


def _create_mcp_client():
    if MCP_AUTH_MODE == "oauth":
        return OauthHttpMCPClient(mcp_server_url="http://localhost:8008/mcp")
    return ApiKeyMCPClient(mcp_server_url="http://localhost:8007/mcp", api_key=MCP_API_KEY)


async def main():
    async with _create_mcp_client() as mcp_client:
        print("\n=== Available Tools ===")
        tools: list[dict] = await mcp_client.get_tools()
        for tool in tools:
            print(json.dumps(tool, indent=2))

        agent = AgentMCPAuth(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            tools=tools,
            mcp_client=mcp_client,
        )

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=DEFAULT_SYSTEM_PROMPT)
        ]

        print(f"MCP-based Agent is ready (auth mode: {MCP_AUTH_MODE})! Type your query or 'exit' to exit.")
        while True:
            try:
                user_input = input("\n> ").strip()
            except EOFError:
                break
            if user_input.lower() == "exit":
                break
            if not user_input:
                continue

            messages.append(Message(role=Role.USER, content=user_input))

            ai_message: Message = await agent.get_completion(messages)
            messages.append(ai_message)


if __name__ == "__main__":
    asyncio.run(main())