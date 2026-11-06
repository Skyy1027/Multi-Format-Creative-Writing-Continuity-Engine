"""Schemas for standup comedy premise, punchline, and callback tracking."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Punchline(BaseModel):
    punchline_id: str = Field(
        description="Short stable slug, e.g. 'pl1_english_chaos' — human-readable"
    )
    text: str = Field(description="The text of the punchline")
    segment_delivered: int = Field(
        description="The segment number where this punchline was delivered"
    )
    attached_to_premise_id: str = Field(
        description="Must reference an existing premise_id (required)"
    )


class Tag(BaseModel):
    text: str = Field(description="The text of the tag/additional punchline")
    segment_introduced: int = Field(
        description="The segment number where this tag appears"
    )
    attached_to_premise_id: str = Field(
        description="Must reference an existing premise_id"
    )


class Premise(BaseModel):
    premise_id: str = Field(
        description='Short stable slug, e.g. "p1_pitstop" — human-readable, not a UUID'
    )
    description: str = Field(description="Description of the premise/setup")
    category: str = Field(
        description='Category: "observational", "absurd_escalation", "character_bit", or "other"'
    )
    status: str = Field(
        description='Status: "open", "resolved", or "callback_used"'
    )
    introduced_in_segment: int = Field(
        description="The segment number where this premise was first introduced"
    )
    callback_segments: List[int] = Field(
        default_factory=list,
        description="List of segment numbers that referenced/resolved this premise"
    )
    resolution_note: Optional[str] = Field(
        default=None,
        description="How the premise was resolved (if resolved)"
    )


class SetState(BaseModel):
    premises: List[Premise] = Field(default_factory=list)
    punchlines: List[Punchline] = Field(default_factory=list)
    tags: List[Tag] = Field(default_factory=list)

