"""Evaluation harness for standup comedy coverage and classification."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.standup_coverage import find_abandoned_setups
from core.standup_extraction import extract_segment_state
from core.standup_schema import SetState

EVAL_DIR = Path("eval")
SEGMENT_DELIMITER = "[SEGMENT]"


def run_coverage_eval():
    """Evaluate abandoned-setup detection against annotated set transcripts."""
    test_cases_path = EVAL_DIR / "test_cases_standup_coverage.json"
    with open(test_cases_path) as f:
        test_cases = json.load(f)

    print(f"\n{'='*70}")
    print("  STANDUP COVERAGE EVALUATION")
    print(f"{'='*70}")
    print(f"Loaded {len(test_cases)} coverage test case(s)\n")

    for tc in test_cases:
        tc_id = tc["id"]
        transcript_path = Path(tc["set_transcript_path"])
        expected_ids = set(tc["expected_abandoned_premise_ids"])

        print(f"{'─'*70}")
        print(f"  Test: {tc_id} — {tc['description']}")
        print(f"{'─'*70}")

        if not expected_ids or not transcript_path.exists():
            print("  [SKIP] Missing test inputs or annotations.")
            continue

        full_text = transcript_path.read_text().strip()
        if not full_text or full_text.startswith("# TODO"):
            print("  [SKIP] Placeholder transcript.")
            continue

        segments = [s.strip() for s in full_text.split(SEGMENT_DELIMITER) if s.strip()]
        state = SetState()
        for seg_num, seg_text in enumerate(segments, start=1):
            state = extract_segment_state(seg_text, seg_num, state)

        abandoned = find_abandoned_setups(state)
        got_ids = {p.premise_id for p in abandoned}

        tp = expected_ids & got_ids
        precision = len(tp) / len(got_ids) if got_ids else (1.0 if not expected_ids else 0.0)
        recall = len(tp) / len(expected_ids) if expected_ids else 1.0

        print(f"  Expected:  {expected_ids}")
        print(f"  Detected:  {got_ids}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall:    {recall:.1%}\n")


def run_classification_eval():
    """Per-segment classification accuracy harness."""
    test_cases_path = EVAL_DIR / "test_cases_standup_classification.json"
    with open(test_cases_path) as f:
        test_cases = json.load(f)

    print(f"\n{'='*70}")
    print("  STANDUP CLASSIFICATION EVALUATION")
    print(f"{'='*70}")
    print(f"Loaded {len(test_cases)} classification test case(s)\n")

    for tc in test_cases:
        tc_id = tc["id"]
        segment_path = Path(tc["segment_text_path"])
        expected = tc["expected_classifications"]

        print(f"{'─'*70}")
        print(f"  Test: {tc_id} — {tc['description']}")
        print(f"{'─'*70}")

        if not expected or not segment_path.exists():
            print("  [SKIP] Awaiting ground truth labels.")
            continue

        segment_text = segment_path.read_text().strip()
        if not segment_text or segment_text.startswith("# TODO"):
            print("  [SKIP] Placeholder segment text.")
            continue

        print("  Ready for comparison against expected labels.")

    print()


if __name__ == "__main__":
    run_coverage_eval()
    run_classification_eval()

