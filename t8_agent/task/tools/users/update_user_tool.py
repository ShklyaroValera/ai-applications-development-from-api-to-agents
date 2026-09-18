from typing import Any

from commons.user_service.user_info import UserUpdate
from t8_agent.task.tools.users.base import BaseUserServiceTool


class UpdateUserTool(BaseUserServiceTool):

    @property
    def name(self) -> str:
        return "update_user"

    @property
    def description(self) -> str:
        return "Updates an existing user in the User Service by user ID with the provided new information."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "number",
                    "description": "User ID that should be updated.",
                },
                "new_info": UserUpdate.model_json_schema(),
            },
            "required": ["id"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            user_id = int(arguments["id"])
            new_info = UserUpdate.model_validate(arguments.get("new_info") or {})
            return self._user_client.update_user(user_id, new_info)
        except Exception as e:
            return f"Error while updating user: {str(e)}"
