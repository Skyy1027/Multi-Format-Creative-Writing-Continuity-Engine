"""Standup comedy segment extraction chain with 4-way joke classification."""

import json
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.llm_config import invoke_with_model_cascade
from core.standup_schema import Premise, Punchline, SetState, Tag


# Intermediate classification schemas returned by the model
class NewPremise(BaseModel):
    premise_id: str = Field(description="Short readable slug, e.g. 'p3_airplane_food'")
    description: str = Field(description="Description of the premise/setup")
    category: str = Field(
        description='Category: "observational", "absurd_escalation", "character_bit", or "other"'
    )


class NewPunchline(BaseModel):
    punchline_id: str = Field(description="Short readable slug, e.g. 'pl3_airplane_peanuts'")
    text: str = Field(description="The text of the punchline")
    attached_to_premise_id: str = Field(
        description="The premise_id this punchline resolves (must be 'open' status)"
    )


class NewTag(BaseModel):
    text: str = Field(description="The text of the tag")
    attached_to_premise_id: str = Field(description="The premise_id this tag piggybacks on")


class CallbackResolution(BaseModel):
    resolves_premise_id: str = Field(
        description="The premise_id of the already-resolved premise being called back"
    )
    resolution_note: str = Field(description="How the premise was referenced or subverted")


class SegmentExtractionResult(BaseModel):
    new_premises: List[NewPremise] = Field(default_factory=list)
    new_punchlines: List[NewPunchline] = Field(default_factory=list)
    new_tags: List[NewTag] = Field(default_factory=list)
    callbacks: List[CallbackResolution] = Field(default_factory=list)


SEGMENT_EXTRACTION_SYSTEM_PROMPT = """\
You are a structural analyst for standup comedy transcripts. You will be \
given the FULL premise list (with their current status — open, resolved, \
or callback_used) as JSON, and the text of a new segment.

For each distinct joke unit in this segment, classify it into EXACTLY ONE \
of these four categories:

1. NEW PREMISE — a setup that creates an expectation not yet established. \
Output it as a new premise with a short readable premise_id (a slug like \
"p4_airport_security", NOT a UUID). Only extract premises with a clear \
setup structure — do not treat every topic mentioned as a premise.

2. PUNCHLINE — the FIRST resolution of a premise. This is the laugh-line \
that delivers on the expectation created by the premise, flipping it from \
"open" to "resolved". A premise should have exactly one punchline. \
Output it referencing that premise's id with a short punchline_id slug. \
A punchline can reference a premise introduced in THIS segment or from \
the premise list, BUT only if the premise's status is currently "open". \
CRITICAL: If the referenced premise is still "open", this is a punchline, \
NOT a callback.

3. TAG — an ADDITIONAL punchline that piggybacks on a premise AFTER its \
initial punchline has already landed, without re-establishing the setup. \
Tags exploit the same comedic framework for extra laughs. Output it \
referencing that premise's id. When in doubt between tag and new premise, \
prefer classifying as a new premise only if it introduces a genuinely \
new expectation; otherwise classify it as a tag.

4. CALLBACK — a reference back to a premise that is ALREADY RESOLVED \
(status "resolved" or "callback_used") from an EARLIER segment. Callbacks \
bring back a previously-closed topic for a bonus laugh. Output the \
matching premise_id, and note how it was referenced. \
CRITICAL RULE: A callback can ONLY target a premise that is already \
"resolved" or "callback_used" — never an "open" premise. If the premise \
you want to reference is still "open", classify it as a PUNCHLINE instead.

Key distinctions:
- PUNCHLINE vs TAG: Punchline is the FIRST payoff. Tags are SUBSEQUENT \
payoffs on the same premise.
- PUNCHLINE vs CALLBACK: Punchline targets an OPEN premise (first \
resolution). Callback targets an ALREADY-RESOLVED premise (bonus laugh).

Respond only with the structured JSON matching the schema — no prose.\
"""


def extract_segment_state(
    segment_text: str,
    segment_number: int,
    existing_state: SetState,
) -> SetState:
    """Classify jokes in segment and merge into cumulative set state."""
    all_premises = [
        {
            "premise_id": p.premise_id,
            "description": p.description,
            "status": p.status,
            "introduced_in_segment": p.introduced_in_segment,
        }
        for p in existing_state.premises
    ]

    human_message = (
        f"Current segment number: {segment_number}\n\n"
        f"Full premise list (ALL statuses — open, resolved, callback_used):\n"
        f"```json\n{json.dumps(all_premises, indent=2)}\n```\n\n"
        f"New segment text:\n\n{segment_text}"
    )

    extraction = invoke_with_model_cascade(SegmentExtractionResult, [
        SystemMessage(content=SEGMENT_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])

    return _merge_segment_extraction(existing_state, extraction, segment_number)


def _merge_segment_extraction(
    existing: SetState,
    extraction: SegmentExtractionResult,
    segment_number: int,
) -> SetState:
    """Merge extracted joke units into SetState, resolving premise statuses."""
    premises = list(existing.premises)
    punchlines = list(existing.punchlines)
    tags = list(existing.tags)
    premise_index = {p.premise_id: i for i, p in enumerate(premises)}

    # 1. New setups
    for new_p in extraction.new_premises:
        premise = Premise(
            premise_id=new_p.premise_id,
            description=new_p.description,
            category=new_p.category,
            status="open",
            introduced_in_segment=segment_number,
            callback_segments=[],
            resolution_note=None,
        )
        premises.append(premise)
        premise_index[new_p.premise_id] = len(premises) - 1

    # 2. Punchlines (first resolution: open -> resolved)
    for new_pl in extraction.new_punchlines:
        punchlines.append(Punchline(
            punchline_id=new_pl.punchline_id,
            text=new_pl.text,
            segment_delivered=segment_number,
            attached_to_premise_id=new_pl.attached_to_premise_id,
        ))

        if new_pl.attached_to_premise_id in premise_index:
            idx = premise_index[new_pl.attached_to_premise_id]
            premise = premises[idx]
            if premise.status == "open":
                premises[idx] = premise.model_copy(update={
                    "status": "resolved",
                    "resolution_note": f"Resolved by punchline in segment {segment_number}",
                })

    # 3. Tags (piggyback laughs, status untouched)
    for new_t in extraction.new_tags:
        tags.append(Tag(
            text=new_t.text,
            segment_introduced=segment_number,
            attached_to_premise_id=new_t.attached_to_premise_id,
        ))

    # 4. Callbacks (reference resolved premise: resolved -> callback_used)
    for cb in extraction.callbacks:
        if cb.resolves_premise_id in premise_index:
            idx = premise_index[cb.resolves_premise_id]
            premise = premises[idx]

            if premise.status in ("resolved", "callback_used"):
                premises[idx] = premise.model_copy(update={
                    "status": "callback_used",
                    "resolution_note": cb.resolution_note,
                    "callback_segments": premise.callback_segments + [segment_number],
                })
            elif premise.status == "open":
                # Fallback: if model tagged an open premise as callback, treat as punchline
                print(
                    f"  [Auto-correct] Callback targets open premise "
                    f"'{cb.resolves_premise_id}' — treating as punchline."
                )
                punchlines.append(Punchline(
                    punchline_id=f"pl_corrected_{cb.resolves_premise_id}",
                    text=cb.resolution_note,
                    segment_delivered=segment_number,
                    attached_to_premise_id=cb.resolves_premise_id,
                ))
                premises[idx] = premise.model_copy(update={
                    "status": "resolved",
                    "resolution_note": f"Resolved by callback in segment {segment_number}",
                })

    return SetState(premises=premises, punchlines=punchlines, tags=tags)


def load_set_state(path: str = "storage/standup_state.json") -> SetState:
    filepath = Path(path)
    if not filepath.exists():
        return SetState()

    content = filepath.read_text().strip()
    if not content or content == "{}":
        return SetState()

    return SetState.model_validate_json(content)


def save_set_state(state: SetState, path: str = "storage/standup_state.json") -> None:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(state.model_dump_json(indent=2))

