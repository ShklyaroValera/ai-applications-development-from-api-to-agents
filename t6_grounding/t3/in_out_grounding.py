import asyncio
import json
import re
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pydantic import BaseModel, Field

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient

# Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# ---
# Implemented:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


SYSTEM_PROMPT = """You are a RAG-powered assistant that groups users by their hobbies.

## Flow:
1. The user asks to find people by hobbies / interests.
2. The most relevant users are retrieved from the vector store.
3. You are provided with `CONTEXT` (retrieved users: user `id` and information about the user) and `USER QUESTION`.
4. You perform Named Entity Extraction: find the concrete hobbies from `CONTEXT` that are relevant to the
   `USER QUESTION` and group the IDs of users who have each hobby.

## Instructions:
- Use ONLY users and IDs present in `CONTEXT`, never invent IDs.
- Put a user ID into a hobby group ONLY if that user's `about_me` explicitly mentions this hobby (or a direct synonym).
  Being retrieved is not enough: most retrieved users are only loosely related, skip them.
- Include a hobby only if it is relevant to the `USER QUESTION`.
- Use short lowercase hobby names (e.g. "hiking", "rock climbing", "camping").
- If no users match, return an empty list.
"""

USER_PROMPT = """## CONTEXT:
{context}

## USER QUESTION:
{query}"""

_BATCH_SIZE = 50

llm_client = OpenAI(api_key=OPENAI_API_KEY)


class GroupingResult(BaseModel):
    hobby: str = Field(description="Hobby. Example: football, painting, hiking, photography, bird watching...")
    user_ids: list[int] = Field(description="IDs of users from the context who have this hobby.")


class GroupingResults(BaseModel):
    grouping_results: list[GroupingResult] = Field(
        description="Users grouped by hobbies relevant to the user question.",
        default_factory=list,
    )


def format_user_document(user: dict[str, Any]) -> str:
    """Embed only `id` and `about_me` - hobbies live in `about_me`, other fields only waste tokens."""
    return f"User:\n  id: {user.get('id')}\n  about_me: {user.get('about_me')}\n"


def _to_document(user: dict[str, Any]) -> Document:
    return Document(id=str(user.get('id')), page_content=format_user_document(user))


class InputGrounder:
    """Adaptive vector based input grounding: Chroma vectorstore synced with User Service on every request."""

    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: OpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.user_client = UserServiceClient()
        self.vectorstore: Optional[Chroma] = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def initialize_vectorstore(self):
        """Cold start: embed all current users."""
        print("🔎 Loading all users for initial vectorstore...")
        users = self.user_client.get_all_users()
        documents = [_to_document(user) for user in users]

        print(f"↗️ Creating embeddings for {len(documents)} users...")
        # cosine space: relevance = 1 - cosine distance (Chroma's default L2 squashes scores below the 0.2 threshold)
        self.vectorstore = Chroma(
            collection_name="users",
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        await self._add_documents(documents)
        print("✅ Vectorstore is ready.")

    async def _add_documents(self, documents: list[Document]):
        if not documents:
            return
        batches = [documents[i:i + _BATCH_SIZE] for i in range(0, len(documents), _BATCH_SIZE)]
        await asyncio.gather(*[self.vectorstore.aadd_documents(batch) for batch in batches])

    async def _update_vectorstore(self):
        """Sync vectorstore with User Service: remove deleted users, embed only new ones."""
        users = self.user_client.get_all_users()
        users_by_id = {str(user.get('id')): user for user in users}

        stored_ids = set(self.vectorstore.get(include=[]).get("ids", []))
        actual_ids = set(users_by_id.keys())

        ids_to_delete = stored_ids - actual_ids
        ids_to_add = actual_ids - stored_ids

        if ids_to_delete:
            self.vectorstore.delete(ids=list(ids_to_delete))
        await self._add_documents([_to_document(users_by_id[user_id]) for user_id in ids_to_add])

        print(f"🔄 Vectorstore synced: +{len(ids_to_add)} new, -{len(ids_to_delete)} deleted users")

    async def retrieve_context(self, query: str, k: int = 100, score: float = 0.2) -> str:
        if self.vectorstore is None:
            await self.initialize_vectorstore()
        else:
            await self._update_vectorstore()

        print("Retrieving context...")
        relevant_docs = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            score_threshold=score,
        )

        context_parts = []
        for doc, relevance_score in relevant_docs:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")
        print(f"{'=' * 100}\n")

        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> GroupingResults:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_prompt},
        ]
        response = self.llm_client.chat.completions.parse(
            model='gpt-4.1-mini',
            temperature=0.0,
            messages=messages,
            response_format=GroupingResults,
        )
        return response.choices[0].message.parsed or GroupingResults()


class OutputGrounder:
    """'Lite' output grounding: verify that every ID returned by LLM exists and fetch the full, original user info."""

    def __init__(self):
        self.user_client = UserServiceClient()

    async def ground_response(self, grouping_results: GroupingResults) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for grouping_result in grouping_results.grouping_results:
            users = await self._find_users(grouping_result.user_ids)
            if users:
                result.setdefault(grouping_result.hobby, []).extend(users)
        return result

    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        async def safe_get_user(user_id: int) -> Optional[dict[str, Any]]:
            try:
                return await self.user_client.get_user(user_id)
            except Exception as e:
                if "404" in str(e):
                    print(f"⚠️ User with ID {user_id} is absent (404), skipped")
                    return None
                raise

        users = await asyncio.gather(*[safe_get_user(user_id) for user_id in dict.fromkeys(ids)])
        return [user for user in users if user is not None]


async def main():
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=OPENAI_API_KEY,
        dimensions=384,
    )
    output_grounder = OutputGrounder()

    async with InputGrounder(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find people who love to watch stars and night sky")
        print(" - I need people to go to fishing together")

        while True:
            try:
                user_question = input("> ").strip()
            except EOFError:
                break
            if not user_question:
                continue
            if user_question.lower() in ['quit', 'exit']:
                break

            context = await rag.retrieve_context(user_question)
            if not context:
                # Nothing relevant retrieved: don't let the LLM invent user IDs
                print("--- No relevant information found ---")
                continue
            augmented_prompt = rag.augment_prompt(user_question, context)
            grouping_results = rag.generate_answer(augmented_prompt)
            print(f"LLM grouping (IDs only): {grouping_results.model_dump_json()}")

            # Output grounding: keep only IDs that really were in the retrieved context
            retrieved_ids = {int(i) for i in re.findall(r"^\s*id: (\d+)$", context, flags=re.MULTILINE)}
            for group in grouping_results.grouping_results:
                group.user_ids = [i for i in group.user_ids if i in retrieved_ids]
            grouping_results.grouping_results = [g for g in grouping_results.grouping_results if g.user_ids]

            grounded_result = await output_grounder.ground_response(grouping_results)
            print(json.dumps(grounded_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
