"""Schemas for screenplay character status and prop continuity tracking."""

from typing import List, Optional
from pydantic import BaseModel, Field


class PropState(BaseModel):
    name: str = Field(description="The prop's name")
    last_seen_location: str = Field(
        description="Where the prop was last seen (e.g. 'kitchen table', 'in John's hand')"
    )
    last_seen_scene: int = Field(
        description="The scene number where this prop was last referenced"
    )
    status: str = Field(
        description='Prop status: "in_use", "discarded", "destroyed", or "offscreen"'
    )


class CharacterPhysicalState(BaseModel):
    name: str = Field(description="The character's name")
    physical_status: str = Field(
        description='Physical condition, e.g. "uninjured", "injured (left shoulder)", "deceased"'
    )
    last_updated_scene: int = Field(
        description="The scene number where this character's state was last changed"
    )
    current_location: str = Field(
        description="Where the character currently is (e.g. 'kitchen', 'exterior - park')"
    )


class ScreenplayState(BaseModel):
    characters: List[CharacterPhysicalState] = Field(default_factory=list)
    props: List[PropState] = Field(default_factory=list)
    current_timeline_position: Optional[str] = Field(
        default=None,
        description='Free text timeline marker, e.g. "Day 2, evening"'
    )


class ScreenplayContinuityFlag(BaseModel):
    issue: str = Field(description="Short description of the continuity error")
    severity: str = Field(description='Severity level: "minor", "moderate", or "major"')
    explanation: str = Field(description="Why this conflicts with established continuity")
    conflicting_source: str = Field(description="Which scene or fact it conflicts with")


class ScreenplayVerificationResult(BaseModel):
    flags: List[ScreenplayContinuityFlag] = Field(default_factory=list)

