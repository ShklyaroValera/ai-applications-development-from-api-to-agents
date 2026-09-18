import asyncio
from typing import Any

from openai import AsyncOpenAI

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match \
the search criteria.

## Instructions:
1. Analyze the user question to understand the search criteria (name, surname, email, hobbies, interests, etc.).
2. Examine each user in the provided context and determine whether they match the criteria.
3. For every matching user return their full details exactly in the original format (do not change any values).
4. If no users match the criteria, respond with exactly: "NO_MATCHES_FOUND" and nothing else.
"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that compiles user search results.

## Instructions:
1. Review all search results collected from different batches of the user database.
2. Combine them and remove duplicates (the same user may appear only once).
3. Present the matching users in a clear, organized manner, keeping their original details unchanged.
4. Answer the user question based only on the provided search results.
"""

USER_PROMPT = """## USERS CONTEXT:
{context}

## SEARCH QUERY:
{query}"""


class TokenTracker:

    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens: list[int] = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self) -> dict:
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens,
        }


llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    result = ""
    for user in context:
        result += "User:\n"
        for key, value in user.items():
            result += f"  {key}: {value}\n"
        result += "\n"
    return result


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    response = await llm_client.chat.completions.create(
        model='gpt-4.1-nano',
        temperature=0.0,
        messages=messages,
    )

    total_tokens = response.usage.total_tokens if response.usage else 0
    token_tracker.add_tokens(total_tokens)

    content = response.choices[0].message.content or ""
    print(f"Response:\n{content}\nTokens used: {total_tokens}\n")
    return content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()

    if not user_question:
        print("Empty question, nothing to search.")
        return

    # 1. Fetch all users and split them into batches of 100
    print("\n--- Searching user database ---")
    users = UserServiceClient().get_all_users()
    user_batches = [users[i:i + 100] for i in range(0, len(users), 100)]

    # 2. Search candidates in every batch IN PARALLEL
    batch_tasks = [
        generate_response(
            system_prompt=BATCH_SYSTEM_PROMPT,
            user_message=USER_PROMPT.format(context=join_context(batch), query=user_question),
        )
        for batch in user_batches
    ]
    batch_results = await asyncio.gather(*batch_tasks)

    # 3. Keep only batches with matches
    print("\n--- Compiling results ---")
    relevant_results = [result for result in batch_results if result.strip() != "NO_MATCHES_FOUND"]

    # 4. Final generation
    print("\n=== SEARCH RESULTS ===")
    if relevant_results:
        combined_results = "\n\n".join(relevant_results)
        await generate_response(
            system_prompt=FINAL_SYSTEM_PROMPT,
            user_message=f"## SEARCH RESULTS:\n{combined_results}\n\n## ORIGINAL QUESTION:\n{user_question}",
        )
    else:
        print(f"No users found matching '{user_question}'")
        print("Try refining your search query.")

    # 5. Performance summary
    summary = token_tracker.get_summary()
    print("\n=== Performance ===")
    print(f"Total API calls: {summary['batch_count']}")
    print(f"Total tokens: {summary['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation