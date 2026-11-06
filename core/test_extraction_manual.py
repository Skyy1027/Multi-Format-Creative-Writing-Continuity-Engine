"""Run sequential extraction over sample chapters and inspect character state."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.extraction import extract_chapter_state, save_lore_state
from core.schema import LoreState

CHAPTERS_DIR = Path("data/sample_chapters")
NUM_CHAPTERS = 4


def main():
    state = LoreState(characters=[])

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
        print(state.model_dump_json(indent=2))
        print(f"\n  → {len(state.characters)} character(s) tracked after chapter {chapter_num}")

    save_lore_state(state)
    print(f"\n{'='*70}")
    print("  State saved to storage/lore_state.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

