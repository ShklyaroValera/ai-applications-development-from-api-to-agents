import os

from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_ENDPOINT, OPENAI_CHAT_COMPLETIONS_ENDPOINT
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t5_rag_advanced.chat.chat_completion_client import ChatCompletionClient
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import TextProcessor, SearchMode

SYSTEM_PROMPT = """You are a RAG-powered assistant that helps users with questions about microwave usage.

## Structure of User message:
Each user message consists of 2 blocks:
- `RAG CONTEXT` - chunks retrieved from the microwave manual that are relevant to the question.
- `USER QUESTION` - the user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` (and the conversation history) to answer the `USER QUESTION`.
- Answer ONLY based on `RAG CONTEXT` and conversation history, never use external knowledge.
- If the question is not related to microwave usage, or the answer is not present in `RAG CONTEXT` or \
conversation history, politely state that you cannot answer this question.
- Be concise and precise.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""

_MANUAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings', 'microwave_manual.txt')

# Embeddings provider. Default: OpenAI 'text-embedding-3-small'.
# Optional (README "Experiment 2"): set env var T5_EMBEDDINGS_PROVIDER=ollama to use local Ollama
# 'nomic-embed-text' (requires `docker compose --profile ollama up -d` in this folder).
if os.getenv('T5_EMBEDDINGS_PROVIDER', 'openai').lower() == 'ollama':
    embeddings_client = EmbeddingsClient(
        endpoint="http://localhost:11434/v1/embeddings",
        model_name='nomic-embed-text',
        api_key="ollama",
    )
else:
    embeddings_client = EmbeddingsClient(
        endpoint=OPENAI_EMBEDDINGS_ENDPOINT,
        model_name='text-embedding-3-small',
        api_key=OPENAI_API_KEY,
    )

completion_client = ChatCompletionClient(
    endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    model_name='gpt-5.2',
    api_key=OPENAI_API_KEY,
)

text_processor = TextProcessor(
    embeddings_client=embeddings_client,
    db_config={
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres',
    },
)


def main():
    print("🎯 Microwave RAG Assistant")
    print("=" * 100)

    load_context = input("\nLoad context to VectorDB (y/n)? > ").strip().lower()
    if load_context in ('y', 'yes'):
        text_processor.process_text_file(
            file_name=_MANUAL_PATH,
            chunk_size=300,
            overlap=40,
            dimensions=384,
        )
        print("=" * 100)

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    while True:
        try:
            user_request = input("\n➡️ ").strip()
        except EOFError:
            print("\n👋 Goodbye")
            break

        if not user_request:
            continue
        if user_request.lower() in ('quit', 'exit'):
            print("👋 Goodbye")
            break

        # Step 1: Retrieval
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        context = text_processor.search(
            search_mode=SearchMode.EUCLIDIAN_DISTANCE,
            user_request=user_request,
            top_k=5,
            score_threshold=0.01,
            dimensions=384,
        )

        # Step 2: Augmentation
        print(f"\n{'=' * 100}\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")
        augmented_prompt = USER_PROMPT.format(context="\n\n".join(context), query=user_request)
        conversation.add_message(Message(Role.USER, augmented_prompt))
        print(f"Prompt:\n{augmented_prompt}")

        # Step 3: Generation
        print(f"\n{'=' * 100}\n🤖 STEP 3: GENERATION\n{'-' * 100}")
        ai_message = completion_client.get_completion(conversation.get_messages())
        print(f"✅ RESPONSE:\n{ai_message.content}")
        print("=" * 100)
        conversation.add_message(ai_message)


# NOTE: Postgres with pgvector must be running on port 5433 -> `docker compose up -d` in t5_rag_advanced/
if __name__ == '__main__':
    main()
