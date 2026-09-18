import json
import anthropic
from pathlib import Path

from commons.constants import ANTHROPIC_API_KEY


SKILLS_VERSION = "skills-2025-10-02"

def _field(obj, *names):
    """anthropic SDK >= 1.x models renamed fields (display_title -> display_name, latest_version -> latest_version_id),
    while the API still returns the old ones - read whichever is present."""
    for name in names:
        value = getattr(obj, name, None)
        if value:
            return value
    return None


def get_or_create_skill(skill_title: str, skill_dir: Path,  client: anthropic.Anthropic) -> str:
    skills = client.beta.skills.list(source="custom", betas=[SKILLS_VERSION])
    for skill in skills.data:
        if _field(skill, "display_title", "display_name") == skill_title:
            print(f"Skill already exists: {skill.id} (latest version: {_field(skill, 'latest_version', 'latest_version_id')})")
            return skill.id

    skill = client.beta.skills.create(
        display_name=skill_title,  # anthropic SDK >= 1.x: `display_title` was renamed to `display_name`
        files=anthropic.lib.files_from_dir(str(skill_dir)),
        betas=[SKILLS_VERSION],
    )
    print(f"Skill uploaded: {skill.id}")
    return skill.id

def delete_skills(client: anthropic.Anthropic):
    skills = client.beta.skills.list(source="custom", betas=[SKILLS_VERSION])
    for skill in skills.data:
        versions = client.beta.skills.versions.list(skill.id, betas=[SKILLS_VERSION])
        for version in versions.data:
            client.beta.skills.versions.delete(_field(version, "version", "id"), skill_id=skill.id, betas=[SKILLS_VERSION])
            print(f"Deleted version {_field(version, 'version', 'id')} of skill '{_field(skill, 'display_title', 'display_name')}'")
        client.beta.skills.delete(skill_id=skill.id, betas=[SKILLS_VERSION])
        print(f"Deleted skill '{_field(skill, 'display_title', 'display_name')}' ({skill.id})")

def chat(client: anthropic.Anthropic, skill_id: str, log_request: bool=True, log_response: bool = True):
    """Multi-turn chat loop that reuses the container across turns."""
    messages = []
    container_id = None
    print("\nStyle Guide Agent is ready. Ask it to write, rewrite, or review any text.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        container = {
            "skills": [
                {
                    "type": "custom",
                    "skill_id": skill_id,
                    "version": "latest",
                }
            ]
        }
        if container_id:
            container["id"] = container_id

        request_payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "messages": messages,
            "container": container,
            "betas": ["code-execution-2025-08-25", SKILLS_VERSION],
            "tools": [
                {
                    "type": "code_execution_20250825",
                    "name": "code_execution",
                }
            ],
        }

        if log_request:
            print("\n--- REQUEST ---")
            print(json.dumps(request_payload, indent=2, default=str))
            print("---------------\n")

        response = client.beta.messages.create(**request_payload)

        if log_response:
            print("\n--- RESPONSE ---")
            print(json.dumps(response.model_dump(), indent=2, default=str))
            print("----------------\n")
        else:
            reply = " ".join(block.text for block in response.content if getattr(block, "type", None) == "text")
            print(f"\nClaude: {reply}\n")

        if getattr(response, "container", None):
            container_id = response.container.id

        messages.append({"role": "assistant", "content": response.content})




STYLE_SKILL_TITLE = "style-guide"
STYLE_SKILL_DIR = Path(__file__).parent / "_skills" / STYLE_SKILL_TITLE

CALCULATOR_SKILL_TITLE = "calculator"
CALCULATOR_SKILL_DIR = Path(__file__).parent / "_skills" / CALCULATOR_SKILL_TITLE

def main():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    # To test the calculator skill, switch to CALCULATOR_SKILL_TITLE / CALCULATOR_SKILL_DIR
    skill_id = get_or_create_skill(
        skill_title=STYLE_SKILL_TITLE,
        skill_dir=STYLE_SKILL_DIR,
        client=client,
    )
    try:
        chat(client, skill_id)
    finally:
        delete_skills(client)


if __name__ == "__main__":
    main()