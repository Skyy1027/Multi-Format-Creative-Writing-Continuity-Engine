"""Manual test script to process scenes sequentially and track screenplay state."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.screenplay_extraction import extract_scene_state, save_screenplay_state
from core.screenplay_schema import ScreenplayState

SCENES_DIR = Path("data/sample_screenplay")
NUM_SCENES = 5


def main():
    state = ScreenplayState()

    for scene_num in range(1, NUM_SCENES + 1):
        scene_path = SCENES_DIR / f"scene_{scene_num:02d}.txt"
        if not scene_path.exists():
            print(f"[SKIP] {scene_path} not found.")
            continue

        scene_text = scene_path.read_text().strip()
        if not scene_text or scene_text.startswith("# TODO"):
            print(f"[SKIP] {scene_path.name} is a placeholder.")
            continue

        print(f"\n{'='*70}")
        print(f"  Processing scene {scene_num}: {scene_path.name}")
        print(f"{'='*70}")

        state = extract_scene_state(scene_text, scene_num, state)
        print(state.model_dump_json(indent=2))
        print(
            f"\n  → {len(state.characters)} character(s), "
            f"{len(state.props)} prop(s) tracked after scene {scene_num}"
        )

    save_screenplay_state(state)
    print(f"\n{'='*70}")
    print("  State saved to storage/screenplay_state.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

