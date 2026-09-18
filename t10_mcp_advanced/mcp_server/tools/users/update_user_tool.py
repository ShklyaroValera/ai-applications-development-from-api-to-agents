from typing import Any

from commons.user_service.user_info import UserUpdate
from t10_mcp_advanced.mcp_server.tools.users.base import BaseUserServiceTool


class UpdateUserTool(BaseUserServiceTool):

    @property
    def name(self) -> str:
        return "update_user"

    @property
    def description(self) -> str:
        return "Updates user info by user `id`. Only fields provided in `new_info` are changed"

    @property
    def input_schema(self) -> dict[str, Any]:
        new_info_schema = UserUpdate.model_json_schema()
        # Nested pydantic models are referenced as `#/$defs/...`, which is resolved from the ROOT of the schema,
        # so `$defs` must be moved to the top level, otherwise LLM providers reject the schema.
        defs = new_info_schema.pop("$defs", None)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "number",
                    "description": "User ID that should be updated."
                },
                "new_info": new_info_schema
            },
            "required": ["id"]
        }
        if defs:
            schema["$defs"] = defs
        return schema

    async def execute(self, arguments: dict[str, Any]) -> str:
        user_id = int(arguments["id"])
        user_update = UserUpdate.model_validate(arguments.get("new_info") or {})
        # UserServiceClient is synchronous (requests-based), so no await here
        return self._user_client.update_user(user_id, user_update)
