import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from commons.constants import OPENAI_API_KEY

# Paths are resolved relative to this file, so the app works both from the repo root
# (`python -m t4_rag_fundamentals.app`) and from inside the folder.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MANUAL_PATH = os.path.join(_BASE_DIR, 'microwave_manual.txt')
_INDEX_PATH = os.path.join(_BASE_DIR, 'microwave_faiss_index')

_SYSTEM_PROMPT = """You are a RAG-powered assistant that helps users with questions about microwave usage.

## Structure of User message:
The user message consists of 2 blocks:
- `RAG CONTEXT` - information retrieved from the microwave manual that is relevant to the user's question.
- `USER QUESTION` - the user's actual question.

## Instructions:
- Answer the `USER QUESTION` using ONLY information from `RAG CONTEXT` and the conversation history.
- Do NOT use any external or prior knowledge.
- If `RAG CONTEXT` is empty or does not contain information relevant to the question, or the question is not \
about the microwave, strictly refuse: say that you cannot answer this question based on the available context.
- Be concise and precise; when helpful, reference the relevant part of the context.
"""

_USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """
        Load existing FAISS index from disk or create a new one.
        Returns:
              VectorStore: Initialized FAISS vectorstore.
        """
        print("🔄 Initializing Microwave Manual RAG System...")

        if os.path.exists(_INDEX_PATH):
            vectorstore = FAISS.load_local(
                folder_path=_INDEX_PATH,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("✅ Loaded existing FAISS index")
        else:
            vectorstore = self._create_new_index()
            print("✅ RAG system initialized successfully!")

        return vectorstore

    def _create_new_index(self) -> VectorStore:
        """
        Load the manual, split into chunks, embed, and save a new FAISS index.
        Returns:
              VectorStore: Newly created and saved FAISS vectorstore.
        """
        print("📖 Loading text document...")
        documents = TextLoader(_MANUAL_PATH, encoding='utf-8').load()

        print("✂️ Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        print("🔍 Creating embeddings and FAISS index...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(_INDEX_PATH)
        print(f"💾 Index saved to {_INDEX_PATH}")

        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3):
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score >= {score}:")

        relevant_docs = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            score_threshold=score,
        )

        context_parts = []
        for doc, relevance_score in relevant_docs:
            context_parts.append(doc.page_content)
            print(f"\n--- (Relevance Score: {relevance_score:.3f}) ---")
            print(f"Content: {doc.page_content}")

        print("=" * 100)
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str):
        """
        Inject retrieved context and user query into the prompt template.
        Args:
              query (str): The user's question.
              context (str): Retrieved context from the vectorstore.
        Returns:
              str: Formatted prompt ready for the LLM.
        """
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = _USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str):
        """
        Send the augmented prompt to the LLM and return its response.
        Args:
              augmented_prompt (str): The prompt with injected context and query.
        Returns:
              str: The LLM-generated answer.
        """
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)

        print(f"{response.content}\n{'=' * 100}")
        return response.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")
    print("Ask a question about the microwave (type 'exit' to quit).")

    while True:
        try:
            user_question = input("\n> ").strip()
        except EOFError:
            break
        if not user_question:
            continue
        if user_question.lower() in ('exit', 'quit'):
            break

        # Step 1: Retrieval (play with `k` and `score` params here)
        context = rag.retrieve_context(user_question)
        # Step 2: Augmentation
        augmented_prompt = rag.augment_prompt(user_question, context)
        # Step 3: Generation
        rag.generate_answer(augmented_prompt)


if __name__ == '__main__':
    main(
        MicrowaveRAG(
            embeddings=OpenAIEmbeddings(
                model='text-embedding-3-small',
                api_key=SecretStr(OPENAI_API_KEY),
            ),
            llm_client=ChatOpenAI(
                temperature=0.0,
                model='gpt-5.2',
                api_key=SecretStr(OPENAI_API_KEY),
            ),
        )
    )
