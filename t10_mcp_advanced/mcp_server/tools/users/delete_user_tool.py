from typing import Any

from t10_mcp_advanced.mcp_server.tools.users.base import BaseUserServiceTool


class DeleteUserTool(BaseUserServiceTool):

    @property
    def name(self) -> str:
        # Named `delete_user` (singular) to be consistent with other tools and the reference solution
        return "delete_user"

    @property
    def description(self) -> str:
        return "Deletes user by user `id` from the Users Management Service"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "number",
                    "description": "ID of the user that should be deleted"
                }
            },
            "required": ["id"]
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        user_id = int(arguments["id"])
        # UserServiceClient is synchronous (requests-based), so no await here
        return self._user_client.delete_user(user_id)
