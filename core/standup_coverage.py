"""Deterministic coverage checks and density metrics for standup sets (no LLM)."""

from typing import Dict, List
from core.standup_schema import Premise, SetState


def find_abandoned_setups(set_state: SetState) -> List[Premise]:
    """Premises still marked 'open' that were never resolved or called back."""
    return [p for p in set_state.premises if p.status == "open"]


def find_premises_without_punchlines(set_state: SetState) -> List[Premise]:
    """Premises with no associated punchline payoff."""
    premises_with_punchlines = {
        pl.attached_to_premise_id for pl in set_state.punchlines
    }
    return [
        p for p in set_state.premises
        if p.premise_id not in premises_with_punchlines
    ]


def compute_laugh_density(set_state: SetState) -> Dict:
    """Compute laugh density (punchlines + tags) per premise and overall."""
    per_premise: Dict[str, Dict[str, int]] = {
        p.premise_id: {"punchlines": 0, "tags": 0, "total": 0}
        for p in set_state.premises
    }

    for pl in set_state.punchlines:
        if pl.attached_to_premise_id in per_premise:
            per_premise[pl.attached_to_premise_id]["punchlines"] += 1
            per_premise[pl.attached_to_premise_id]["total"] += 1

    for tag in set_state.tags:
        if tag.attached_to_premise_id in per_premise:
            per_premise[tag.attached_to_premise_id]["tags"] += 1
            per_premise[tag.attached_to_premise_id]["total"] += 1

    total_premises = len(set_state.premises)
    total_punchlines = len(set_state.punchlines)
    total_tags = len(set_state.tags)
    total_laughs = total_punchlines + total_tags
    overall_density = total_laughs / total_premises if total_premises > 0 else 0.0

    return {
        "per_premise": per_premise,
        "overall_density": overall_density,
        "total_premises": total_premises,
        "total_punchlines": total_punchlines,
        "total_tags": total_tags,
    }

