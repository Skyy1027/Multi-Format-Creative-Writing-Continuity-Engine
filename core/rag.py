"""Vector storage and semantic retrieval pipeline using ChromaDB."""

from pathlib import Path
from typing import List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SentenceTransformerEmbedding(chromadb.EmbeddingFunction):
    """Embedding wrapper for ChromaDB using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


_embedding_fn: Optional[SentenceTransformerEmbedding] = None


def _get_embedding_fn() -> SentenceTransformerEmbedding:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbedding()
    return _embedding_fn


def get_vector_store(
    persist_directory: str = "storage/chroma_db",
    collection_name: str = "chapters",
):
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_get_embedding_fn(),
    )


def ingest_chapter(
    chapter_text: str,
    chapter_number: int,
    vector_store,
) -> int:
    """Split chapter text into chunks and store with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(chapter_text)
    ids = [f"ch{chapter_number:02d}_chunk{i:03d}" for i in range(len(chunks))]
    metadatas = [{"chapter_number": chapter_number} for _ in chunks]

    vector_store.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas,
    )
    return len(chunks)


def retrieve_relevant_context(
    query_text: str,
    vector_store,
    k: int = 5,
    exclude_chapter: Optional[int] = None,
) -> List[str]:
    """Retrieve top-k relevant chunks, optionally excluding a specific chapter."""
    if exclude_chapter is not None:
        where_filter = {"chapter_number": {"$ne": exclude_chapter}}
        results = vector_store.query(
            query_texts=[query_text],
            n_results=k,
            where=where_filter,
        )
    else:
        results = vector_store.query(
            query_texts=[query_text],
            n_results=k,
        )

    if results and results.get("documents"):
        return results["documents"][0]
    return []

