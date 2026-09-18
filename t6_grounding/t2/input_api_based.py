from enum import StrEnum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient

QUERY_ANALYSIS_PROMPT = """You are a query analysis system that extracts search parameters from user questions.

## Available search fields:
- `name` - user's first name
- `surname` - user's last name
- `email` - user's email address

## Instructions:
1. Analyze the user question and identify explicit values that can be used for search.
2. Map every extracted value to the appropriate search field.
3. Extract ONLY values that are clearly stated in the question - do not infer, guess or assume values.
4. If there are no explicit values for the available fields, return an empty list of parameters.

## Examples:
- "Who is John?" -> name: "John"
- "Find John Smith" -> name: "John", surname: "Smith"
- "Find users with surname Adams" -> surname: "Adams"
- "Who has email john.smith@example.com?" -> email: "john.smith@example.com"
- "I need users that love hiking" -> no parameters
"""

SYSTEM_PROMPT = """You are a RAG-powered assistant that helps to find information about users.

## Structure of User message:
- `RAG CONTEXT` - users retrieved from the User Service that are relevant to the question.
- `USER QUESTION` - the user's actual question.

## Instructions:
- Answer ONLY based on the provided `RAG CONTEXT` and conversation history.
- If no relevant information exists in `RAG CONTEXT`, state that you cannot answer the question.
- When presenting user information, format it clearly (e.g. a list with the key fields of each user).
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION:
{query}"""


class SearchField(StrEnum):
    NAME = "name"
    SURNAME = "surname"
    EMAIL = "email"


class SearchRequest(BaseModel):
    search_field: SearchField = Field(description="Search field")
    search_value: str = Field(description="Search value. Sample: Adam.")


class SearchRequests(BaseModel):
    search_request_parameters: list[SearchRequest] = Field(
        description="List of search parameters to execute",
        default_factory=list
    )


llm_client = OpenAI(api_key=OPENAI_API_KEY)

user_client = UserServiceClient()


def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": QUERY_ANALYSIS_PROMPT},
        {"role": "user", "content": user_question},
    ]
    response = llm_client.chat.completions.parse(
        model='gpt-4.1-nano',
        temperature=0.0,
        messages=messages,
        response_format=SearchRequests,
    )

    parsed: SearchRequests | None = response.choices[0].message.parsed
    search_parameters = parsed.search_request_parameters if parsed else []

    if search_parameters:
        request_params = {param.search_field.value: param.search_value for param in search_parameters}
        print(f"Searching with parameters: {request_params}")
        return user_client.search_users(**request_params)

    print("No specific search parameters found!")
    return []


def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    context_str = ""
    for user in context:
        context_str += "User:\n"
        for key, value in user.items():
            context_str += f"  {key}: {value}\n"
        context_str += "\n"

    augmented_prompt = USER_PROMPT.format(context=context_str, query=user_question)
    print(f"Augmented prompt:\n{augmented_prompt}")
    return augmented_prompt


def generate_answer(augmented_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": augmented_prompt},
    ]
    response = llm_client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.0,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    while True:
        try:
            user_question = input("> ").strip()
        except EOFError:
            break
        if user_question:
            if user_question.lower() in ['quit', 'exit']:
                break

            print("\n--- Retrieving context ---")
            context = retrieve_context(user_question)
            if context:
                print("\n--- Augmenting prompt ---")
                augmented_prompt = augment_prompt(user_question, context)
                print("\n--- Generating answer ---")
                answer = generate_answer(augmented_prompt)
                print(f"\nAnswer: {answer}\n")
            else:
                print("\n--- No relevant information found ---")


if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce