"""State extraction for screenplay scenes (characters and props)."""

from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_config import invoke_with_model_cascade
from core.screenplay_schema import ScreenplayState

SCENE_EXTRACTION_SYSTEM_PROMPT = """\
You are a script supervisor's continuity assistant. You will be given the \
current known state of all characters' physical status/location and all \
tracked props as JSON, and the text of a new scene.

Update the state to reflect this scene:
- Update a character's physical_status only when the scene explicitly shows \
or states a change (injury, change of appearance, etc.).
- Update current_location when a character is shown in a new location.
- Update prop status/location when a prop is picked up, moved, discarded, \
or destroyed.
- If a character or prop is not mentioned in this scene, carry their state \
forward unchanged.
- Set last_updated_scene (for characters) or last_seen_scene (for props) \
to the current scene number ONLY for entries that actually changed or \
appeared in this scene.
- If a new character or prop appears for the first time, add them.
- Update current_timeline_position if the scene establishes a time \
(e.g. "Day 2, morning"). If no time is stated, carry forward the \
previous value unchanged.
- Do not infer injuries or prop states that are not clearly shown or \
stated in the scene text.

Respond only with the structured JSON matching the schema — no prose.\
"""


def extract_scene_state(
    scene_text: str,
    scene_number: int,
    existing_state: ScreenplayState,
) -> ScreenplayState:
    """Extract and merge character physical status and prop states from a scene."""
    existing_state_json = existing_state.model_dump_json(indent=2)
    human_message = (
        f"Current scene number: {scene_number}\n\n"
        f"Current known state:\n```json\n{existing_state_json}\n```\n\n"
        f"New scene text:\n\n{scene_text}"
    )

    return invoke_with_model_cascade(ScreenplayState, [
        SystemMessage(content=SCENE_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])


def load_screenplay_state(
    path: str = "storage/screenplay_state.json",
) -> ScreenplayState:
    filepath = Path(path)
    if not filepath.exists():
        return ScreenplayState()

    content = filepath.read_text().strip()
    if not content or content == "{}":
        return ScreenplayState()

    return ScreenplayState.model_validate_json(content)


def save_screenplay_state(
    state: ScreenplayState,
    path: str = "storage/screenplay_state.json",
) -> None:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(state.model_dump_json(indent=2))

