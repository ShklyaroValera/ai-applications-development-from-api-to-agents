import re
from typing import Any

from t8_agent.task.tools.users.base import BaseUserServiceTool


class SearchUsersTool(BaseUserServiceTool):

    @property
    def name(self) -> str:
        return "search_users"

    @property
    def description(self) -> str:
        return (
            "Searches users in the User Service by name, surname, email and/or gender. "
            "All parameters are optional; provided ones are combined as filters."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "surname": {"type": "string", "description": "User surname"},
                "email": {"type": "string", "description": "User email"},
                "gender": {"type": "string", "description": "User gender (e.g. male, female)"},
            },
            "required": [],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            result = self._user_client.search_users(**arguments)
        except Exception as e:
            return f"Error while searching users: {str(e)}"
        gender = (arguments.get("gender") or "").strip().lower()
        return self._filter_by_gender(result, gender) if gender else result

    @staticmethod
    def _filter_by_gender(result: str, gender: str) -> str:
        # The current mock User Service ignores the `gender` query param and returns everyone,
        # so the filter is applied here on the formatted user blocks.
        blocks = re.findall(r"```\n.*?```\n", result, flags=re.S)
        matched = [b for b in blocks if re.search(rf"^\s*gender: {re.escape(gender)}\s*$", b, flags=re.M | re.I)]
        return f"Found {len(matched)} users with gender '{gender}':\n" + "".join(matched)
