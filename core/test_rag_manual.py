"""Manual test script to ingest sample chapters and test RAG retrieval."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag import get_vector_store, ingest_chapter, retrieve_relevant_context

CHAPTERS_DIR = Path("data/sample_chapters")
TEST_CHROMA_DIR = "storage/chroma_db_test_rag"


def main():
    if Path(TEST_CHROMA_DIR).exists():
        shutil.rmtree(TEST_CHROMA_DIR)

    vector_store = get_vector_store(persist_directory=TEST_CHROMA_DIR)

    for chapter_num in range(1, 4):
        chapter_path = CHAPTERS_DIR / f"chapter_{chapter_num:02d}.txt"
        text = chapter_path.read_text()
        n_chunks = ingest_chapter(text, chapter_num, vector_store)
        print(f"Ingested chapter {chapter_num}: {n_chunks} chunks")

    print(f"\n{'='*70}")
    print("  RAG Retrieval Tests")
    print(f"{'='*70}")

    test_queries = [
        ("Kael drew Dawnfang and fought with the enchanted longsword",
         "Should retrieve passages about Dawnfang being traded away in ch3"),
        ("They arrived from the western sea road after a coastal ferry",
         "Should retrieve passages about traveling east on mountain road in ch3"),
        ("Kael used his left arm freely to climb and pull Liora up",
         "Should retrieve passages about left arm injury/sling from ch2-3"),
    ]

    for query, expected in test_queries:
        print(f"\n{'─'*70}")
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"{'─'*70}")

        results = retrieve_relevant_context(query, vector_store, k=3)
        for i, chunk in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"  {chunk[:200]}...")


if __name__ == "__main__":
    main()

