from typing import Any

import requests

from commons.constants import OPENAI_RESPONSES_ENDPOINT
from t8_agent.task.tools.base import BaseTool


class WebSearchTool(BaseTool):

    def __init__(self, open_ai_api_key: str):
        self.__api_key = f"Bearer {open_ai_api_key}"
        self.__endpoint = OPENAI_RESPONSES_ENDPOINT

    @property
    def name(self) -> str:
        return "web_search_tool"

    @property
    def description(self) -> str:
        return (
            "Tool for WEB searching. Use it to find up-to-date public information on the internet "
            "(e.g. facts about a person or company) when it is not available in the User Service."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for on the web",
                }
            },
            "required": ["request"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        # https://developers.openai.com/api/docs/guides/tools-web-search
        headers = {
            "Authorization": self.__api_key,
            "Content-Type": "application/json",
        }
        request_data = {
            "model": "gpt-5.2",
            "tools": [{"type": "web_search"}],
            "input": str(arguments.get("request", "")),
        }

        try:
            response = requests.post(url=self.__endpoint, headers=headers, json=request_data)
        except Exception as e:
            return f"Error while performing web search: {str(e)}"

        if response.status_code != 200:
            return f"Error: {response.status_code} {response.text}"

        data = response.json()
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        return block.get("text", "")
        return "No result returned from web search."
