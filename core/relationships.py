"""Relationship extraction and graph state management."""

from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_config import invoke_with_model_cascade
from core.schema import LoreState, RelationshipSet

RELATIONSHIP_SYSTEM_PROMPT = """\
You are a relationship-extraction assistant for a novel's continuity \
system. You will be given a list of already-known relationships as JSON, \
a list of known character names, and the text of a new chapter.

Identify relationships between characters, and between characters and \
locations, that are stated or clearly implied in this chapter. Use short, \
consistent relation_type labels (prefer reusing a label already present \
in the known relationships list over inventing a near-duplicate, e.g. \
always use "ally of" rather than sometimes "allied with").

Rules:
- If a new chapter shows a relationship has changed (e.g. former allies \
become enemies), do not delete the old relationship — instead add a new \
relationship with status "active" and mark the prior one's status as \
"superseded" in your output.
- If an existing relationship is simply reaffirmed this chapter with no \
change, update its chapter_last_confirmed to the current chapter number \
and leave status as "active".
- Only extract relationships with clear textual support — do not infer \
relationships from weak implication.
- Use the character names exactly as they appear in the known characters \
list to avoid creating duplicate nodes (e.g. always "Kael", not \
"the sellsword").

Respond only with the structured JSON matching the schema — no prose.\
"""


def extract_relationships(
    chapter_text: str,
    chapter_number: int,
    existing_relationships: RelationshipSet,
    lore_state: LoreState,
) -> RelationshipSet:
    """Extract character and location relationships, merging with existing graph."""
    existing_json = existing_relationships.model_dump_json(indent=2)
    known_characters = [c.name for c in lore_state.characters]

    human_message = (
        f"Current chapter number: {chapter_number}\n\n"
        f"Known character names: {known_characters}\n\n"
        f"Current known relationships:\n```json\n{existing_json}\n```\n\n"
        f"New chapter text:\n\n{chapter_text}"
    )

    llm_result = invoke_with_model_cascade(RelationshipSet, [
        SystemMessage(content=RELATIONSHIP_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])

    return _merge_relationships(existing_relationships, llm_result)


def _merge_relationships(
    existing: RelationshipSet,
    llm_output: RelationshipSet,
) -> RelationshipSet:
    """Merge newly extracted relationships into existing state by (source, target, type)."""
    merged = {
        (r.source, r.target, r.relation_type): r
        for r in existing.relationships
    }

    # Overwrite existing or insert newly discovered
    for rel in llm_output.relationships:
        key = (rel.source, rel.target, rel.relation_type)
        merged[key] = rel

    return RelationshipSet(relationships=list(merged.values()))


def load_relationships(
    path: str = "storage/relationships.json",
) -> RelationshipSet:
    filepath = Path(path)
    if not filepath.exists():
        return RelationshipSet(relationships=[])

    content = filepath.read_text().strip()
    if not content or content == "{}":
        return RelationshipSet(relationships=[])

    return RelationshipSet.model_validate_json(content)


def save_relationships(
    relationship_set: RelationshipSet,
    path: str = "storage/relationships.json",
) -> None:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(relationship_set.model_dump_json(indent=2))

