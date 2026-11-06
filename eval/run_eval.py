"""Evaluation harness for narrative continuity verification."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.extraction import extract_chapter_state
from core.rag import get_vector_store, ingest_chapter
from core.schema import LoreState
from core.verification import retrieve_relevant_context, verify_chapter

EVAL_DIR = Path("eval")
CHAPTERS_DIR = Path("data/sample_chapters")
EVAL_CHROMA_DIR = "storage/chroma_db_eval"


def build_state_up_to(chapter_number: int) -> LoreState:
    """Build cumulative lore state for chapters 1..(chapter_number - 1)."""
    state = LoreState(characters=[])
    for ch in range(1, chapter_number):
        chapter_path = CHAPTERS_DIR / f"chapter_{ch:02d}.txt"
        if not chapter_path.exists():
            print(f"  [WARN] {chapter_path} not found, skipping.")
            continue
        state = extract_chapter_state(chapter_path.read_text(), ch, state)
        print(f"  Built state through chapter {ch}")
    return state


def build_vector_store_up_to(chapter_number: int):
    """Ingest history chapters into an isolated test collection."""
    chroma_dir = f"{EVAL_CHROMA_DIR}_ch{chapter_number}"
    if Path(chroma_dir).exists():
        shutil.rmtree(chroma_dir)

    vs = get_vector_store(persist_directory=chroma_dir)
    for ch in range(1, chapter_number):
        chapter_path = CHAPTERS_DIR / f"chapter_{ch:02d}.txt"
        if not chapter_path.exists():
            continue
        n = ingest_chapter(chapter_path.read_text(), ch, vs)
        print(f"  Ingested chapter {ch}: {n} chunks")
    return vs


def run_eval():
    test_cases_path = EVAL_DIR / "test_cases.json"
    with open(test_cases_path) as f:
        test_cases = json.load(f)

    print(f"\nLoaded {len(test_cases)} test cases\n")

    results = []
    state_cache = {}
    vs_cache = {}

    for tc in test_cases:
        tc_id = tc["id"]
        draft_ch = tc["draft_chapter_number"]
        expected = tc["expected_contradiction"]
        description = tc["description"]

        print(f"{'='*70}")
        print(f"  Test: {tc_id} | Draft Ch: {draft_ch} | Expected Flag: {expected}")
        print(f"  {description}")
        print(f"{'='*70}")

        if draft_ch not in state_cache:
            print(f"\n  Building lore state up to chapter {draft_ch - 1}...")
            state_cache[draft_ch] = build_state_up_to(draft_ch)

        if draft_ch not in vs_cache:
            print(f"\n  Building vector store up to chapter {draft_ch - 1}...")
            vs_cache[draft_ch] = build_vector_store_up_to(draft_ch)

        lore_state = state_cache[draft_ch]
        vector_store = vs_cache[draft_ch]
        draft_text = Path(tc["draft_text_path"]).read_text()

        retrieved = retrieve_relevant_context(
            query_text=draft_text,
            vector_store=vector_store,
            k=5,
            exclude_chapter=draft_ch,
        )

        verification = verify_chapter(draft_text, lore_state, retrieved)
        has_flags = len(verification.flags) > 0
        passed = (expected and has_flags) or (not expected and not has_flags)

        print(f"  Flags found: {len(verification.flags)}")
        for flag in verification.flags:
            print(f"    - [{flag.severity.upper()}] {flag.issue}")
            print(f"      {flag.explanation}")
            print(f"      Conflicts with: {flag.conflicting_source}")
        print(f"  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}\n")

        results.append({
            "id": tc_id,
            "expected": expected,
            "has_flags": has_flags,
            "num_flags": len(verification.flags),
            "passed": passed,
            "description": description,
        })

    # Summary table
    print(f"\n{'='*70}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*70}\n")
    print(f"  {'ID':<8} {'Expected':<12} {'Got Flags':<12} {'# Flags':<10} {'Result':<8}")
    print(f"  {'─'*52}")
    for r in results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        print(f"  {r['id']:<8} {str(r['expected']):<12} {str(r['has_flags']):<12} {r['num_flags']:<10} {status:<8}")

    tp = len([r for r in results if r["expected"] and r["has_flags"]])
    fn = len([r for r in results if r["expected"] and not r["has_flags"]])
    tn = len([r for r in results if not r["expected"] and not r["has_flags"]])
    fp = len([r for r in results if not r["expected"] and r["has_flags"]])

    total_pos = tp + fn
    total_neg = tn + fp
    recall = tp / total_pos if total_pos > 0 else 0
    fpr = fp / total_neg if total_neg > 0 else 0

    print(f"\n  Recall:              {recall:.1%} ({tp}/{total_pos})")
    print(f"  False Positive Rate: {fpr:.1%} ({fp}/{total_neg})")
    print(f"  Passed:              {sum(1 for r in results if r['passed'])}/{len(results)}\n")


if __name__ == "__main__":
    run_eval()

