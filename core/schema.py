"""Schemas for character state, relationships, and continuity verification."""

from typing import List
from pydantic import BaseModel, Field


class CharacterState(BaseModel):
    name: str = Field(description="The character's full name")
    current_location: str = Field(description="Where the character currently is")
    inventory: List[str] = Field(
        default_factory=list,
        description="Items the character currently possesses"
    )
    physical_status: str = Field(
        description='Physical condition, e.g. "healthy", "injured (left arm)", "deceased"'
    )
    last_updated_chapter: int = Field(
        description="The chapter number where this character's state was last changed"
    )


class LoreState(BaseModel):
    characters: List[CharacterState] = Field(default_factory=list)


class Relationship(BaseModel):
    source: str = Field(description="Entity name (character or location)")
    target: str = Field(description="Entity name (character or location)")
    relation_type: str = Field(
        description='Short label, e.g. "ally of", "enemy of", "loyal to", '
                    '"located in", "owns", "hired by"'
    )
    chapter_introduced: int = Field(
        default=0,
        description="Chapter where this relationship first appears"
    )
    chapter_last_confirmed: int = Field(
        default=0,
        description="Most recent chapter that still supports this relationship"
    )
    status: str = Field(
        default="active",
        description='Relationship status: "active", "ended", or "superseded"'
    )


class RelationshipSet(BaseModel):
    relationships: List[Relationship] = Field(default_factory=list)


class ContinuityFlag(BaseModel):
    issue: str = Field(description="Short description of the contradiction")
    severity: str = Field(description='Severity level: "minor", "moderate", or "major"')
    explanation: str = Field(description="Why this conflicts with established lore")
    conflicting_source: str = Field(description="Which chapter or fact it conflicts with")


class VerificationResult(BaseModel):
    flags: List[ContinuityFlag] = Field(default_factory=list)

