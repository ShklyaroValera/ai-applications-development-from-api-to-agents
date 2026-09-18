import asyncio
import json
import os

from commons.constants import OPENAI_API_KEY
from commons.models.message import Message
from commons.models.role import Role
from t10_mcp_advanced.agent.agent import CustomAgentMCP
from t10_mcp_advanced.agent.clients.custom_mcp_client import CustomMCPClient
from t10_mcp_advanced.agent.clients.mcp_client import MCPClient


UMS_MCP_URL = "http://localhost:8006/mcp"
# README fetch server https://remote.mcpservers.org/fetch/mcp no longer resolves (NXDOMAIN).
# Exa hosted MCP (no key needed) gives the agent web search + page fetch: web_search_exa, web_fetch_exa.
FETCH_MCP_URL = os.getenv("FETCH_MCP_URL", "https://mcp.exa.ai/mcp")

# Which MCP client implementation to use for BOTH servers (env var `MCP_CLIENT`):
#   sdk    -> MCPClient (official `mcp` SDK), README step 2.1 (default)
#   custom -> CustomMCPClient (pure aiohttp JSON-RPC implementation), README step 2.3-2.4
MCP_CLIENT = os.getenv("MCP_CLIENT", "sdk").lower()

SYSTEM_PROMPT = """You are an advanced AI agent. Your goal is to assist the user with his questions and requests.
You have tools of the Users Management Service (search, get, add, update, delete users) and a `fetch` tool that
downloads web pages, so you can look up public information in the web (e.g. fetch a Wikipedia page).
When asked to add a person, first check whether the user already exists; if information is missing, look it up
with the fetch tool and fill in only the fields you can reasonably derive (don't invent sensitive data).
Before deleting users ask for confirmation. Keep answers concise and structured."""


async def _create_client(url: str) -> MCPClient | CustomMCPClient:
    if MCP_CLIENT == "custom":
        return await CustomMCPClient.create(url)
    if MCP_CLIENT == "sdk":
        return await MCPClient.create(url)
    raise ValueError(f"Unknown MCP_CLIENT `{MCP_CLIENT}`. Use `sdk` or `custom`")


async def _collect_tools(
        client: MCPClient | CustomMCPClient,
        tools: list[dict],
        tool_name_client_map: dict[str, MCPClient | CustomMCPClient],
) -> None:
    for tool in await client.get_tools():
        tool_name = tool["function"]["name"]
        if tool_name in tool_name_client_map:
            print(f"Tool `{tool_name}` is already provided by another MCP server, skipping")
            continue
        tools.append(tool)
        tool_name_client_map[tool_name] = client
        print(json.dumps(tool, indent=2))


async def _close_client(client: MCPClient | CustomMCPClient) -> None:
    try:
        if isinstance(client, CustomMCPClient):
            await client.close()
        else:
            if client._session_context:
                await client._session_context.__aexit__(None, None, None)
            if client._streams_context:
                await client._streams_context.__aexit__(None, None, None)
    except Exception as e:
        print(f"Error while closing MCP client: {e}")


async def main():
    tools: list[dict] = []
    tool_name_client_map: dict[str, MCPClient | CustomMCPClient] = {}
    clients: list[MCPClient | CustomMCPClient] = []

    try:
        for url in (UMS_MCP_URL, FETCH_MCP_URL):
            client = await _create_client(url)
            clients.append(client)
            await _collect_tools(client, tools, tool_name_client_map)

        agent = CustomAgentMCP(
            api_key=OPENAI_API_KEY,
            model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
            tools=tools,
            tool_name_client_map=tool_name_client_map,
        )

        messages: list[Message] = [Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)]

        print(f"MCP-based Agent is ready (MCP_CLIENT={MCP_CLIENT})! Type your query or 'exit' to exit.")
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
            ai_message: Message = await agent.get_completion(messages)
            messages.append(ai_message)
    finally:
        for client in reversed(clients):
            await _close_client(client)


if __name__ == "__main__":
    asyncio.run(main())


# Check if Arkadiy Dobkin present as a user, if not then search info about him in the web and add him