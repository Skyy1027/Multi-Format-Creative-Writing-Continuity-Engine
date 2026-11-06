"""Manual test script to process comedy segments and run coverage check."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.standup_coverage import compute_laugh_density, find_abandoned_setups, find_premises_without_punchlines
from core.standup_extraction import extract_segment_state, save_set_state
from core.standup_schema import SetState

SET_FILE = Path("data/sample_standup/set_01.txt")
SEGMENT_DELIMITER = "[SEGMENT]"


def main():
    if not SET_FILE.exists():
        print(f"[ERROR] {SET_FILE} not found.")
        return

    full_text = SET_FILE.read_text().strip()
    if not full_text or full_text.startswith("# TODO"):
        print(f"[SKIP] {SET_FILE} is empty or a placeholder.")
        return

    segments = [s.strip() for s in full_text.split(SEGMENT_DELIMITER) if s.strip()]
    print(f"Found {len(segments)} segment(s) in {SET_FILE}")

    state = SetState()

    for seg_num, segment_text in enumerate(segments, start=1):
        print(f"\n{'='*70}")
        print(f"  Processing segment {seg_num}")
        print(f"{'='*70}")

        state = extract_segment_state(segment_text, seg_num, state)
        print(state.model_dump_json(indent=2))
        print(
            f"\n  → {len(state.premises)} premise(s), "
            f"{len(state.punchlines)} punchline(s), "
            f"{len(state.tags)} tag(s) after segment {seg_num}"
        )

    # Coverage analysis
    print(f"\n{'='*70}")
    print("  COVERAGE ANALYSIS")
    print(f"{'='*70}")

    abandoned = find_abandoned_setups(state)
    if abandoned:
        print(f"\n  ⚠️ {len(abandoned)} abandoned setup(s):")
        for p in abandoned:
            print(f"    - [{p.premise_id}] {p.description}")
    else:
        print("\n  ✅ No abandoned setups — all premises resolved.")

    unpaid = find_premises_without_punchlines(state)
    if unpaid:
        print(f"\n  ⚠️ {len(unpaid)} premise(s) without a punchline:")
        for p in unpaid:
            print(f"    - [{p.premise_id}] {p.description}")
    else:
        print("\n  ✅ All premises have at least one punchline.")

    density = compute_laugh_density(state)
    print(
        f"\n  Overall laugh density: {density['overall_density']:.2f} "
        f"({density['total_punchlines']} punchlines + "
        f"{density['total_tags']} tags / {density['total_premises']} premises)"
    )
    print("\n  Per-premise breakdown:")
    for pid, counts in density["per_premise"].items():
        print(f"    - {pid}: {counts['punchlines']} punchlines, {counts['tags']} tags = {counts['total']} total")

    save_set_state(state)
    print(f"\n{'='*70}")
    print("  State saved to storage/standup_state.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

