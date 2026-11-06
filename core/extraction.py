"""Structured state extraction chain for chapter text."""

from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_config import invoke_with_model_cascade
from core.schema import LoreState

EXTRACTION_SYSTEM_PROMPT = """\
You are a worldbuilding continuity assistant. You will be given the current \
known state of all characters as JSON, and the text of a new chapter.

Update the character state to reflect what happens in this chapter:
- Track location changes.
- Track inventory changes (items gained or lost).
- Track physical status changes (injuries, healing, death, etc.).
- If a character is mentioned for the first time, add them.
- Do NOT remove characters who are absent from this chapter — carry their \
last known state forward unchanged.
- Set last_updated_chapter to the current chapter number ONLY for characters \
who actually changed or appeared in this chapter.

Respond only with the structured JSON matching the schema — no prose.\
"""


def extract_chapter_state(
    chapter_text: str,
    chapter_number: int,
    existing_state: LoreState,
) -> LoreState:
    """Extract and merge character state from a chapter using Gemini."""
    existing_state_json = existing_state.model_dump_json(indent=2)
    human_message = (
        f"Current chapter number: {chapter_number}\n\n"
        f"Current known character state:\n```json\n{existing_state_json}\n```\n\n"
        f"New chapter text:\n\n{chapter_text}"
    )

    return invoke_with_model_cascade(LoreState, [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])


def load_lore_state(path: str = "storage/lore_state.json") -> LoreState:
    filepath = Path(path)
    if not filepath.exists():
        return LoreState(characters=[])

    content = filepath.read_text().strip()
    if not content or content == "{}":
        return LoreState(characters=[])

    return LoreState.model_validate_json(content)


def save_lore_state(state: LoreState, path: str = "storage/lore_state.json") -> None:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(state.model_dump_json(indent=2))

