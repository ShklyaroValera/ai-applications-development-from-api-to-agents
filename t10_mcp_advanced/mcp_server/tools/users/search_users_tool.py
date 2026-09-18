from typing import Any

from t10_mcp_advanced.mcp_server.tools.users.base import BaseUserServiceTool


class SearchUsersTool(BaseUserServiceTool):

    @property
    def name(self) -> str:
        return "search_users"

    @property
    def description(self) -> str:
        return (
            "Searches users by name, surname, email and gender. Matching for name, surname and email is partial "
            "and case-insensitive. All parameters are optional and can be combined."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User first name (partial match)"
                },
                "surname": {
                    "type": "string",
                    "description": "User last name (partial match)"
                },
                "email": {
                    "type": "string",
                    "description": "User email (partial match)"
                },
                "gender": {
                    "type": "string",
                    "description": "User gender (exact match): male, female, other, prefer_not_to_say"
                },
            },
            "required": []
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        # UserServiceClient is synchronous (requests-based), so no await here
        return self._user_client.search_users(**arguments)
