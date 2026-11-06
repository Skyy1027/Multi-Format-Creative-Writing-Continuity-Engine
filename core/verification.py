"""Continuity verification chain for novel drafts."""

from typing import List
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_config import invoke_with_model_cascade
from core.rag import get_vector_store, retrieve_relevant_context
from core.schema import LoreState, VerificationResult

VERIFICATION_SYSTEM_PROMPT = """\
You are a continuity editor for a novel. You are given:
1. The current known state of all characters as JSON (including relationships).
2. Relevant excerpts from earlier chapters.
3. A new chapter draft.

Carefully check the draft for contradictions with the established character \
state or the earlier excerpts. Look for:

- A character using or referencing an item they no longer have (lost, gave \
away, traded, destroyed in an earlier chapter).
- A character appearing in a location inconsistent with where the story last \
placed them, with no travel explained or with contradictory travel details.
- A character's physical state contradicting an established injury, \
disability, or death (e.g., freely using a limb that was described as \
badly injured and in a sling).
- A relationship between characters contradicting what was previously \
established (e.g., characters acting as strangers when they were allies, \
or a mentor-student dynamic reversing without explanation).
- Any other factual contradictions with established lore.

Only flag genuine contradictions — not stylistic choices, ambiguous cases, \
or things that could reasonably be explained. Be thorough but precise.

For each contradiction found, provide:
- issue: a short description of the contradiction
- severity: "minor", "moderate", or "major"
- explanation: why this conflicts with established lore
- conflicting_source: which specific chapter and fact it conflicts with

If there are no contradictions, return an empty flags list.

Respond only with the structured JSON matching the schema — no prose.\
"""


def verify_chapter(
    draft_text: str,
    lore_state: LoreState,
    retrieved_context: List[str],
) -> VerificationResult:
    """Check a chapter draft against current lore and retrieved context."""
    lore_json = lore_state.model_dump_json(indent=2)
    context_text = (
        "\n\n---\n\n".join(retrieved_context)
        if retrieved_context
        else "(No earlier excerpts available)"
    )

    human_message = (
        f"CURRENT CHARACTER STATE:\n```json\n{lore_json}\n```\n\n"
        f"RELEVANT EXCERPTS FROM EARLIER CHAPTERS:\n\n{context_text}\n\n"
        f"NEW CHAPTER DRAFT:\n\n{draft_text}"
    )

    return invoke_with_model_cascade(VerificationResult, [
        SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])


def verify_chapter_with_rag(
    draft_text: str,
    lore_state: LoreState,
    exclude_chapter: int = None,
    vector_store=None,
    k: int = 5,
) -> VerificationResult:
    """Fetch relevant context from vector store and verify continuity."""
    if vector_store is None:
        vector_store = get_vector_store()

    retrieved = retrieve_relevant_context(
        query_text=draft_text,
        vector_store=vector_store,
        k=k,
        exclude_chapter=exclude_chapter,
    )
    return verify_chapter(draft_text, lore_state, retrieved)

