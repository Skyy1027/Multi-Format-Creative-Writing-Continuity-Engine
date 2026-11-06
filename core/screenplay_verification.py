"""Continuity verification chain for screenplay scenes."""

from typing import List
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_config import invoke_with_model_cascade
from core.rag import get_vector_store, retrieve_relevant_context
from core.screenplay_schema import ScreenplayState, ScreenplayVerificationResult

SCREENPLAY_VERIFICATION_SYSTEM_PROMPT = """\
You are a script continuity checker. You are given:
1. The current known state of character physical status/location and props \
as JSON.
2. Relevant excerpts from earlier scenes.
3. A new scene draft.

Check for continuity errors:
- A character's physical state being inconsistent with an established \
injury or condition without acknowledgment (e.g. performing physical \
actions inconsistent with a stated injury).
- A prop appearing in a scene without explanation when it was last \
recorded as discarded, destroyed, or elsewhere.
- A character appearing in a location inconsistent with the established \
timeline or their last known position without a transition.
- Timeline inconsistencies (e.g. scene set in the "morning" when the \
previous scene was already "evening" of the same day with no time skip).

Only flag clear, textually supported inconsistencies — not stylistic \
choices or ambiguous cases. Be thorough but precise.

For each continuity error found, provide:
- issue: a short description of the error
- severity: "minor", "moderate", or "major"
- explanation: why this conflicts with established continuity
- conflicting_source: which specific scene or fact it conflicts with

If there are no continuity errors, return an empty flags list.

Respond only with the structured JSON matching the schema — no prose.\
"""


def verify_scene(
    scene_text: str,
    screenplay_state: ScreenplayState,
    retrieved_context: List[str],
) -> ScreenplayVerificationResult:
    """Check a scene draft against established character and prop continuity."""
    state_json = screenplay_state.model_dump_json(indent=2)
    context_text = (
        "\n\n---\n\n".join(retrieved_context)
        if retrieved_context
        else "(No earlier excerpts available)"
    )

    human_message = (
        f"CURRENT SCREENPLAY STATE:\n```json\n{state_json}\n```\n\n"
        f"RELEVANT EXCERPTS FROM EARLIER SCENES:\n\n{context_text}\n\n"
        f"NEW SCENE DRAFT:\n\n{scene_text}"
    )

    return invoke_with_model_cascade(ScreenplayVerificationResult, [
        SystemMessage(content=SCREENPLAY_VERIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])


def verify_scene_with_rag(
    scene_text: str,
    screenplay_state: ScreenplayState,
    exclude_scene: int = None,
    vector_store=None,
    k: int = 5,
) -> ScreenplayVerificationResult:
    """Fetch relevant context from vector store and verify scene continuity."""
    if vector_store is None:
        vector_store = get_vector_store()

    # Pass scene_number through exclude_chapter param in rag
    retrieved = retrieve_relevant_context(
        query_text=scene_text,
        vector_store=vector_store,
        k=k,
        exclude_chapter=exclude_scene,
    )
    return verify_scene(scene_text, screenplay_state, retrieved)

