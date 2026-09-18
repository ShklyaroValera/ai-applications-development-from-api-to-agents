from pathlib import Path
from typing import Any

from t12_skills.custom.file_utils import get_file_content
from t12_skills.custom.tools.base import BaseTool


class ReadSkillTool(BaseTool):
    """Reads files from the local skills directory by path."""

    def __init__(self, skills_dir: Path):
        self._skills_dir = skills_dir.resolve()

    @property
    def name(self) -> str:
        return "read_skill"

    @property
    def description(self) -> str:
        return (
            "Read a skill file by its path. Use it to load a skill's instructions (SKILL.md) before "
            "performing a task, and to read any referenced skill resources on demand (scripts, "
            "references, examples, assets). Paths are relative to the skills root and start with "
            "the skill name, e.g. /unit-converter/SKILL.md or /unit-converter/scripts/convert.py"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the skill file relative to the skills root, starting with the skill name. "
                        "E.g. /unit-converter/SKILL.md or /unit-converter/examples.md"
                    ),
                }
            },
            "required": ["path"],
        }

    async def _execute(self, arguments: dict[str, Any]) -> str:
        relative_path = str(arguments["path"]).lstrip("/")
        full_path = (self._skills_dir / relative_path).resolve()
        if not full_path.is_relative_to(self._skills_dir):
            return f"ERROR: Path is outside of the skills directory: {arguments['path']}"
        return get_file_content(full_path)