import os

from commons.constants import OPENAI_API_KEY, ANTHROPIC_API_KEY
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from commons.user_service.client import UserServiceClient

from t8_agent.task.agents.anthropic import AnthropicBasedAgent
from t8_agent.task.agents.openai import OpenAIBasedAgent
from t8_agent.task.prompts import SYSTEM_PROMPT
from t8_agent.task.tools.users.create_user_tool import CreateUserTool
from t8_agent.task.tools.users.delete_user_tool import DeleteUserTool
from t8_agent.task.tools.users.get_user_by_id_tool import GetUserByIdTool
from t8_agent.task.tools.users.search_users_tool import SearchUsersTool
from t8_agent.task.tools.users.update_user_tool import UpdateUserTool
from t8_agent.task.tools.web_search import WebSearchTool

# Which LLM provider drives the agent: "openai" (default) or "anthropic".
# Switch without editing code: `AGENT_PROVIDER=anthropic`.
AGENT_PROVIDER = os.getenv("AGENT_PROVIDER", "openai").strip().lower()


def main():
    user_client = UserServiceClient()
    tools = [
        WebSearchTool(open_ai_api_key=OPENAI_API_KEY),
        GetUserByIdTool(user_client),
        SearchUsersTool(user_client),
        CreateUserTool(user_client),
        UpdateUserTool(user_client),
        DeleteUserTool(user_client),
    ]

    if AGENT_PROVIDER == "anthropic":
        agent = AnthropicBasedAgent(
            model="claude-sonnet-4-5",
            api_key=ANTHROPIC_API_KEY,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
    else:
        agent = OpenAIBasedAgent(
            model="gpt-5.2",
            api_key=OPENAI_API_KEY,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

    conversation = Conversation()

    print("Type your question or 'exit' to quit.")
    print("Samples:")
    print("  Add Andrej Karpathy as a new user")
    print("  Find all female users")
    print("  Delete user with id 3")

    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            print("\nExiting the chat. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        conversation.add_message(Message(role=Role.USER, content=user_input))

        ai_message = agent.get_response(conversation.get_messages(), print_request=True)
        conversation.add_message(ai_message)
        print("🤖:", ai_message.content)
        print("=" * 100)
        print()


main()
