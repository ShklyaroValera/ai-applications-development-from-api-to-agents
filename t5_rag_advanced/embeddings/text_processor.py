import os
from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: EmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(
            self,
            file_name: str,
            chunk_size: int,
            overlap: int,
            dimensions: int,
            truncate_table: bool = True,
    ):
        """
        Load content from file, split it into chunks, generate embeddings and save them to DB.
        Args:
            file_name: path to the text file
            chunk_size: size of each chunk in chars (min 10)
            overlap: number of chars overlapping between neighbour chunks
            dimensions: embedding dimensions (must match the `vectors.embedding` column, 384)
            truncate_table: truncate `vectors` table before inserting
        """
        if chunk_size < 10:
            raise ValueError("chunk_size must be at least 10")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be lower than chunk_size")

        if truncate_table:
            self._truncate_table()

        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = chunk_text(content, chunk_size, overlap)
        embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)

        print(f"Processing document: {file_name}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total embeddings: {len(embeddings)}")

        document_name = os.path.basename(file_name)
        for index, chunk in enumerate(chunks):
            self._save_chunk(embeddings[index], chunk, document_name)

    def _truncate_table(self):
        """Remove all rows from the `vectors` table"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors")
            conn.commit()
        print("Table `vectors` has been truncated.")

    def _save_chunk(self, embedding: list[float], chunk: str, document_name: str):
        """Insert a chunk with its embedding into the `vectors` table"""
        vector_string = self._to_vector_string(embedding)

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (document_name, chunk, vector_string),
                )
            conn.commit()

        print(f"Stored chunk from document: {document_name}")

    def search(
            self,
            search_mode: SearchMode,
            user_request: str,
            top_k: int,
            score_threshold: float,
            dimensions: int,
    ) -> list[str]:
        """
        Perform similarity search in the `vectors` table.
        Args:
            search_mode: distance metric (cosine or euclidean)
            user_request: user query
            top_k: max number of chunks to return
            score_threshold: min similarity score, range 0.0..1.0
            dimensions: embedding dimensions (must be the same as the data persisted in the VectorDB)
        Returns:
            list of retrieved chunk texts, most relevant first
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if score_threshold < 0 or score_threshold > 1:
            raise ValueError("score_threshold must be in [0.0..1.0] range")

        query_embedding = self.embeddings_client.get_embeddings(inputs=user_request, dimensions=dimensions)[0]
        vector_string = self._to_vector_string(query_embedding)

        # Convert similarity score threshold into the max allowed distance for the chosen metric:
        #   cosine:    similarity = 1 - distance          -> distance <= 1 - score
        #   euclidean: similarity = 1 / (1 + distance)    -> distance <= 1/score - 1
        if search_mode == SearchMode.COSINE_DISTANCE:
            operator = '<=>'
            max_distance = 1.0 - score_threshold
        else:
            operator = '<->'
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        query = f"""SELECT text, embedding {operator} %s::vector AS distance
                    FROM vectors
                    WHERE embedding {operator} %s::vector <= %s
                    ORDER BY distance
                    LIMIT %s"""

        retrieved_chunks = []
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (vector_string, vector_string, max_distance, top_k))
                rows = cursor.fetchall()

        for row in rows:
            distance = float(row['distance'])
            similarity = 1.0 - distance if search_mode == SearchMode.COSINE_DISTANCE else 1.0 / (1.0 + distance)
            print(f"---Similarity score: {similarity:.3f} (distance: {distance:.3f})---")
            print(f"Data: {row['text']}\n")
            retrieved_chunks.append(row['text'])

        return retrieved_chunks

    @staticmethod
    def _to_vector_string(embedding: list[float]) -> str:
        """pgvector literal, e.g. '[0.1,0.2,0.3]' (cast with ::vector in SQL)"""
        return f"[{','.join(map(str, embedding))}]"
