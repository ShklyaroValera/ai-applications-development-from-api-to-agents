import json
import uuid
from typing import Optional, Any
import aiohttp


MCP_SESSION_ID_HEADER = "Mcp-Session-Id"

class CustomMCPClient:
    """Pure Python MCP client without external MCP libraries"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session_id: Optional[str] = None
        self.http_session: Optional[aiohttp.ClientSession] = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'CustomMCPClient':
        """Async factory method to create and connect CustomMCPClient"""
        instance = cls(mcp_server_url)
        await instance.connect()
        return instance

    async def _send_request(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Send JSON-RPC request to MCP server"""
        if not self.http_session:
            raise RuntimeError("HTTP session not initialized")

        request_data: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
        }
        if params:
            request_data["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if method != "initialize" and self.session_id:
            headers[MCP_SESSION_ID_HEADER] = self.session_id

        async with self.http_session.post(self.server_url, json=request_data, headers=headers) as response:
            if not self.session_id and response.headers.get(MCP_SESSION_ID_HEADER):
                self.session_id = response.headers[MCP_SESSION_ID_HEADER]

            if response.status == 202:
                return {}

            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type.lower():
                response_data = await self._parse_sse_response_streaming(response)
            else:
                response_data = await response.json(content_type=None)

            if "error" in response_data:
                error = response_data["error"]
                raise RuntimeError(f"MCP Error {error['code']}: {error['message']}")

            return response_data


    async def _parse_sse_response_streaming(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        """Parse Server-Sent Events response with streaming"""
        # Response stream sample:
        # data: {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "some tool call result"}]}}
        # data: [DONE]
        async for line in response.content:
            line_str = line.decode("utf-8").strip()

            if not line_str or line_str.startswith(":"):
                continue

            if line_str.startswith("data:"):
                data_part = line_str[5:].strip()  # remove 'data:' prefix (with or without space)
                if data_part and data_part != "[DONE]":
                    # we need only the first chunk since MCP tool returns response with 1 chunk
                    return json.loads(data_part)

        raise RuntimeError("No valid data found in SSE response")

    async def connect(self) -> None:
        """Connect to MCP server and initialize session"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        self.http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)

        try:
            init_params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "my-custom-mcp-client", "version": "1.0.0"},
            }
            init_result = await self._send_request("initialize", init_params)
            await self._send_notification("notifications/initialized")
            print(f"Capabilities: {json.dumps(init_result, indent=2)}")
        except Exception as e:
            await self.close()
            raise RuntimeError(f"Failed to connect to MCP server: {e}")

    async def _send_notification(self, method: str) -> None:
        """Send notification (no response expected)"""
        if self.http_session is None:
            raise RuntimeError("HTTP session not initialized")

        request_data = {
            "jsonrpc": "2.0",
            "method": method,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers[MCP_SESSION_ID_HEADER] = self.session_id

        async with self.http_session.post(self.server_url, json=request_data, headers=headers) as response:
            if MCP_SESSION_ID_HEADER in response.headers:
                self.session_id = response.headers[MCP_SESSION_ID_HEADER]
                print(f"Session ID: {self.session_id}")

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools from MCP server"""
        if not self.http_session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        response = await self._send_request("tools/list")
        tools = response.get("result", {}).get("tools", [])
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for tool in tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server"""
        if self.http_session is None:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        print(f"    Calling `{tool_name}` with {tool_args}")

        params = {
            "name": tool_name,
            "arguments": tool_args,
        }
        response = await self._send_request("tools/call", params)

        if content := response.get("result", {}).get("content", []):
            if item := content[0]:
                text_result = item.get("text", "")
                print(f"    ⚙️: {text_result}\n")
                return text_result

        return "Unexpected error occurred!"

    async def close(self) -> None:
        """Close underlying HTTP session"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        self.http_session = None