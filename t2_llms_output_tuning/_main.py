import copy
import json
import os
from typing import Any

from t2_llms_output_tuning._clients._base_client import AIClient
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role


def run(
        client: AIClient,
        print_request: bool = True,
        print_only_content: bool = False,
        **kwargs
) -> None:
    conversation = Conversation()

    print("Type your question or 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
    
        if user_input.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break
    
        conversation.add_message(Message(Role.USER, user_input))

        print("AI:")
        ai_message = client.response(
            messages=conversation.get_messages(),
            print_request=print_request,
            print_only_content=print_only_content,
            **kwargs
        )
        conversation.add_message(ai_message)


def select_experiment(
        experiments: dict[str, tuple[str, dict[str, Any], str]],
        default: str,
) -> tuple[str, dict[str, Any]]:
    """
    Pick one experiment preset by name from the T2_EXPERIMENT env var (falls back to `default`).
    Each preset is (model_name, request_kwargs, suggested_query). T2_MODEL env var overrides the preset model.
    Returns (model_name, request_kwargs).
    """
    name = os.getenv("T2_EXPERIMENT", default).strip() or default
    if name not in experiments:
        raise ValueError(f"Unknown T2_EXPERIMENT '{name}'. Available: {', '.join(experiments)}")

    model_name, kwargs, query = experiments[name]
    model_name = os.getenv("T2_MODEL", model_name).strip() or model_name

    print(f"Available experiments (set T2_EXPERIMENT=<name>): {', '.join(experiments)}")
    print(f"Experiment: '{name}' | model: {model_name} | params: {json.dumps(kwargs)}")
    print(f"Suggested query: {query}\n")
    return model_name, copy.deepcopy(kwargs)
