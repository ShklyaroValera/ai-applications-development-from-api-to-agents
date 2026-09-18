from contextlib import AsyncExitStack
from typing import Any

from mcp.types import Prompt, Resource
from pydantic import AnyUrl

from t9_mcp_fundamentals.agent.mcp_clients.base import MCPClient


class MultiMCPClient(MCPClient):
    """
    OPTIONAL task: aggregates several MCP clients (1 client <-> 1 server) behind the MCPClient interface.

    Tools from all servers are exposed to the Agent as a single list, and `call_tool` routes each call
    to the client that owns the tool (tool name -> client registry), so the Agent itself stays unchanged.
    """

    def __init__(self, clients: list[MCPClient]) -> None:
        super().__init__()
        self.clients = clients
        self._tool_registry: dict[str, MCPClient] = {}
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self):
        self._exit_stack = AsyncExitStack()
        for client in self.clients:
            await self._exit_stack.enter_async_context(client)
        # Mark as connected (base class checks `self.session`); use the first connected session
        self.session = self.clients[0].session if self.clients else None
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def get_tools(self) -> list[dict[str, Any]]:
        all_tools: list[dict[str, Any]] = []
        self._tool_registry.clear()
        for client in self.clients:
            for tool in await client.get_tools():
                name = tool["function"]["name"]
                if name in self._tool_registry:
                    print(f"Tool name collision: `{name}` is already provided by another MCP server, skipping")
                    continue
                self._tool_registry[name] = client
                all_tools.append(tool)
        return all_tools

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        if not self._tool_registry:
            await self.get_tools()
        client = self._tool_registry.get(tool_name)
        if client is None:
            raise ValueError(f"Unknown tool `{tool_name}`: no connected MCP server provides it")
        return await client.call_tool(tool_name, tool_args)

    async def get_resources(self) -> list[Resource]:
        resources: list[Resource] = []
        for client in self.clients:
            resources.extend(await client.get_resources())
        return resources

    async def get_resource(self, uri: AnyUrl) -> str:
        last_error: Exception | None = None
        for client in self.clients:
            try:
                return await client.get_resource(uri)
            except Exception as e:
                last_error = e
        raise ValueError(f"Resource {uri} not found on any MCP server: {last_error}")

    async def get_prompts(self) -> list[Prompt]:
        prompts: list[Prompt] = []
        for client in self.clients:
            prompts.extend(await client.get_prompts())
        return prompts

    async def get_prompt(self, name: str) -> str:
        for client in self.clients:
            if any(p.name == name for p in await client.get_prompts()):
                return await client.get_prompt(name)
        raise ValueError(f"Prompt `{name}` not found on any MCP server")
