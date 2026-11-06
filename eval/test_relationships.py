"""Manual test script to extract and evolve relationships across chapters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.extraction import extract_chapter_state
from core.relationships import extract_relationships, save_relationships
from core.schema import LoreState, RelationshipSet

CHAPTERS_DIR = Path("data/sample_chapters")
NUM_CHAPTERS = 4


def main():
    state = LoreState(characters=[])
    relationships = RelationshipSet(relationships=[])

    for chapter_num in range(1, NUM_CHAPTERS + 1):
        chapter_path = CHAPTERS_DIR / f"chapter_{chapter_num:02d}.txt"
        if not chapter_path.exists():
            print(f"[ERROR] {chapter_path} not found, skipping.")
            continue

        chapter_text = chapter_path.read_text()
        print(f"\n{'='*70}")
        print(f"  Processing chapter {chapter_num}: {chapter_path.name}")
        print(f"{'='*70}")

        state = extract_chapter_state(chapter_text, chapter_num, state)
        print(f"  → {len(state.characters)} character(s) tracked")

        relationships = extract_relationships(
            chapter_text, chapter_num, relationships, state,
        )
        print(f"\n  Relationships after chapter {chapter_num}:")
        print(relationships.model_dump_json(indent=2))

        active = [r for r in relationships.relationships if r.status == "active"]
        superseded = [r for r in relationships.relationships if r.status == "superseded"]
        ended = [r for r in relationships.relationships if r.status == "ended"]
        print(
            f"\n  → {len(active)} active, {len(superseded)} superseded, "
            f"{len(ended)} ended ({len(relationships.relationships)} total)"
        )

    save_relationships(relationships)
    print(f"\n{'='*70}")
    print("  Final relationships saved to storage/relationships.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

