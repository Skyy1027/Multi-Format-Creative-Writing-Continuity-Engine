"""Streamlit UI for the Dynamic Narrative Continuity Engine."""

import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from streamlit_agraph import agraph

# Narrative domain
from core.extraction import extract_chapter_state, load_lore_state, save_lore_state
from core.graph_builder import (
    build_graph_elements,
    build_screenplay_graph_elements,
    build_standup_graph_elements,
    get_graph_config,
)
from core.rag import get_vector_store, ingest_chapter, retrieve_relevant_context
from core.relationships import extract_relationships, load_relationships, save_relationships
from core.verification import verify_chapter

# Screenplay domain
from core.screenplay_extraction import extract_scene_state, load_screenplay_state, save_screenplay_state
from core.screenplay_schema import ScreenplayState
from core.screenplay_verification import verify_scene

# Standup domain
from core.standup_coverage import compute_laugh_density, find_abandoned_setups, find_premises_without_punchlines
from core.standup_extraction import extract_segment_state, load_set_state, save_set_state
from core.standup_schema import SetState

# State storage paths
LORE_STATE_PATH = str(PROJECT_ROOT / "storage" / "lore_state.json")
RELATIONSHIPS_PATH = str(PROJECT_ROOT / "storage" / "relationships.json")
CHROMA_PATH = str(PROJECT_ROOT / "storage" / "chroma_db")

SCREENPLAY_STATE_PATH = str(PROJECT_ROOT / "storage" / "screenplay_state.json")
SCREENPLAY_CHROMA_PATH = str(PROJECT_ROOT / "storage" / "chroma_db_screenplay")

STANDUP_STATE_PATH = str(PROJECT_ROOT / "storage" / "standup_state.json")

st.set_page_config(
    page_title="Narrative Continuity Engine",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state defaults
for key, default in {
    "chapter_number": 1,
    "verification_result": None,
    "process_success": False,
    "scene_number": 1,
    "screenplay_verification_result": None,
    "screenplay_process_success": False,
    "standup_process_success": False,
    "coverage_result": None,
    "standup_processing_log": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.title("🎛️ Domain")
    domain = st.radio(
        "Select domain",
        ["Narrative", "Screenplay", "Standup Comedy"],
        label_visibility="collapsed",
    )
    st.divider()


def render_continuity_flags(flags, empty_message: str):
    """Helper to render detected continuity flags with severity colors."""
    if not flags:
        st.success(empty_message)
        return

    st.warning(f"⚠️ **{len(flags)} continuity issue(s) detected:**")
    severity_map = {
        "major": ("🔴", "red"),
        "moderate": ("🟠", "orange"),
        "minor": ("🟡", "yellow"),
    }

    for flag in flags:
        icon, color = severity_map.get(flag.severity.lower(), ("🟡", "yellow"))
        with st.expander(f"{icon} [{flag.severity.upper()}] {flag.issue}", expanded=True):
            st.markdown(f"**Severity:** :{color}[{flag.severity.upper()}]")
            st.markdown(f"**Issue:** {flag.issue}")
            st.markdown(f"**Explanation:** {flag.explanation}")
            st.markdown(f"**Conflicting Source:** {flag.conflicting_source}")


# ===================================================================
# NARRATIVE DOMAIN
# ===================================================================

if domain == "Narrative":

    with st.sidebar:
        st.title("📋 Character State")
        st.caption("Live state from `storage/lore_state.json`")
        st.divider()

        lore_state = load_lore_state(LORE_STATE_PATH)

        if not lore_state.characters:
            st.info("No characters tracked yet. Process a chapter to begin.")
        else:
            for char in lore_state.characters:
                with st.container():
                    st.subheader(char.name)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Location", char.current_location)
                    with col2:
                        status = char.physical_status
                        if "deceased" in status.lower():
                            st.metric("Status", f"💀 {status}")
                        elif "injured" in status.lower() or "sling" in status.lower():
                            st.metric("Status", f"🤕 {status}")
                        else:
                            st.metric("Status", f"✅ {status}")

                    if char.inventory:
                        st.markdown("**Inventory:**")
                        for item in char.inventory:
                            st.markdown(f"  - {item}")
                    else:
                        st.markdown("**Inventory:** *(empty)*")

                    st.caption(f"Last updated: Chapter {char.last_updated_chapter}")
                    st.divider()

        st.markdown(f"**Next chapter number:** {st.session_state.chapter_number}")

    st.title("📖 Narrative Continuity Engine")
    st.markdown(
        "Process chapters to build your lore state, then verify new drafts "
        "for contradictions with established canon."
    )

    tab_plot_holes, tab_graph, tab_dataset = st.tabs(
        ["🔍 Plot Hole Checker", "🕸️ Knowledge Graph", "📦 Dataset"]
    )

    # --- Plot Hole Checker ---
    with tab_plot_holes:
        draft_text = st.text_area(
            "New Chapter Draft",
            height=400,
            placeholder="Paste or type your chapter text here...",
            key="narrative_draft_input",
        )

        col_process, col_verify = st.columns(2)
        with col_process:
            process_clicked = st.button(
                "⚙️ Process Chapter",
                use_container_width=True,
                type="primary",
                help="Ingest this chapter into the vector store and extract character state.",
            )
        with col_verify:
            verify_clicked = st.button(
                "🔍 Verify Continuity",
                use_container_width=True,
                help="Check this draft against established lore for contradictions.",
            )

        if process_clicked:
            if not draft_text.strip():
                st.error("Please enter chapter text before processing.")
            else:
                try:
                    with st.spinner(f"Processing chapter {st.session_state.chapter_number}..."):
                        vector_store = get_vector_store(persist_directory=CHROMA_PATH)
                        n_chunks = ingest_chapter(
                            draft_text, st.session_state.chapter_number, vector_store,
                        )
                        current_state = load_lore_state(LORE_STATE_PATH)
                        updated_state = extract_chapter_state(
                            draft_text, st.session_state.chapter_number, current_state,
                        )
                        save_lore_state(updated_state, LORE_STATE_PATH)

                        try:
                            existing_rels = load_relationships(RELATIONSHIPS_PATH)
                            updated_rels = extract_relationships(
                                draft_text, st.session_state.chapter_number,
                                existing_rels, updated_state,
                            )
                            save_relationships(updated_rels, RELATIONSHIPS_PATH)
                            rel_count = len(updated_rels.relationships)
                        except Exception as rel_err:
                            st.warning(
                                f"⚠️ Relationship extraction failed (character state saved): {rel_err}"
                            )
                            rel_count = None

                        st.session_state.chapter_number += 1
                        st.session_state.process_success = True
                        st.session_state.verification_result = None

                    msg = (
                        f"✅ Chapter {st.session_state.chapter_number - 1} processed! "
                        f"({n_chunks} chunks ingested, state updated"
                    )
                    if rel_count is not None:
                        msg += f", {rel_count} relationships tracked"
                    msg += ")"
                    st.success(msg)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing chapter: {str(e)}")
                    st.exception(e)

        if verify_clicked:
            if not draft_text.strip():
                st.error("Please enter chapter text before verifying.")
            else:
                try:
                    with st.spinner("Checking for continuity issues..."):
                        current_state = load_lore_state(LORE_STATE_PATH)
                        vector_store = get_vector_store(persist_directory=CHROMA_PATH)
                        retrieved = retrieve_relevant_context(
                            query_text=draft_text, vector_store=vector_store,
                            k=5, exclude_chapter=st.session_state.chapter_number,
                        )
                        result = verify_chapter(draft_text, current_state, retrieved)
                        st.session_state.verification_result = result
                except Exception as e:
                    st.error(f"❌ Error during verification: {str(e)}")
                    st.exception(e)

        if st.session_state.verification_result is not None:
            st.divider()
            render_continuity_flags(
                st.session_state.verification_result.flags,
                "✅ **No continuity issues detected.** The draft is consistent with established lore.",
            )

        if st.session_state.process_success:
            st.session_state.process_success = False

    # --- Knowledge Graph ---
    with tab_graph:
        st.subheader("Interactive Relationship Graph")
        graph_lore = load_lore_state(LORE_STATE_PATH)
        graph_rels = load_relationships(RELATIONSHIPS_PATH)

        if not graph_rels.relationships:
            st.info("No relationships extracted yet — process a chapter in the **Plot Hole Checker** tab first.")
        else:
            nodes, edges = build_graph_elements(graph_lore, graph_rels)
            agraph(nodes=nodes, edges=edges, config=get_graph_config())

            st.caption(
                "🔵 **Circle** = Character  ·  "
                "⬜ **Square** = Location  ·  "
                "⚫ **Gray** = Deceased  ·  "
                "🟣 **Purple** = Other entity  ·  "
                "**Arrows** = Active relationships"
            )

            with st.expander("📜 Show relationship history (including superseded/ended)"):
                grouped = defaultdict(list)
                for rel in graph_rels.relationships:
                    grouped[f"{rel.source} → {rel.target}"].append(rel)

                for pair_key in sorted(grouped.keys()):
                    rels = sorted(grouped[pair_key], key=lambda r: r.chapter_introduced)
                    st.markdown(f"**{pair_key}**")
                    for rel in rels:
                        status_icon = {"active": "🟢", "superseded": "🟡", "ended": "🔴"}.get(rel.status, "⚪")
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;{status_icon} "
                            f"Ch {rel.chapter_introduced}–{rel.chapter_last_confirmed}: "
                            f"**{rel.relation_type}** [{rel.status}]"
                        )
                    st.markdown("---")

    # --- Dataset Export ---
    with tab_dataset:
        st.subheader("LoRA Dataset Export")
        st.info("🚧 This tab allows exporting extracted lore data as fine-tuning datasets for LoRA training.")


# ===================================================================
# SCREENPLAY DOMAIN
# ===================================================================

elif domain == "Screenplay":

    with st.sidebar:
        st.title("🎬 Screenplay State")
        st.caption("Live state from `storage/screenplay_state.json`")
        st.divider()

        sp_state = load_screenplay_state(SCREENPLAY_STATE_PATH)

        if not sp_state.characters and not sp_state.props:
            st.info("No state tracked yet. Process a scene to begin.")
        else:
            if sp_state.current_timeline_position:
                st.markdown(f"**Timeline:** {sp_state.current_timeline_position}")
                st.divider()

            if sp_state.characters:
                st.markdown("### Characters")
                for char in sp_state.characters:
                    with st.container():
                        st.subheader(char.name)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Location", char.current_location)
                        with col2:
                            status = char.physical_status
                            if "deceased" in status.lower():
                                st.metric("Status", f"💀 {status}")
                            elif "injured" in status.lower():
                                st.metric("Status", f"🤕 {status}")
                            else:
                                st.metric("Status", f"✅ {status}")
                        st.caption(f"Last updated: Scene {char.last_updated_scene}")
                        st.divider()

            if sp_state.props:
                st.markdown("### Props")
                status_icons = {"in_use": "🟢", "offscreen": "🟡", "discarded": "🔴", "destroyed": "💥"}
                for prop in sp_state.props:
                    icon = status_icons.get(prop.status, "⚪")
                    st.markdown(
                        f"{icon} **{prop.name}** — "
                        f"{prop.last_seen_location} [{prop.status}] "
                        f"(scene {prop.last_seen_scene})"
                    )
                st.divider()

        st.markdown(f"**Next scene number:** {st.session_state.scene_number}")

    st.title("🎬 Screenplay Continuity Checker")
    st.markdown(
        "Process scenes to track character physical state and props, "
        "then verify new scene drafts for continuity errors."
    )

    tab_continuity, tab_graph = st.tabs(["🔍 Continuity Checker", "🕸️ Knowledge Graph"])

    with tab_continuity:
        scene_text = st.text_area(
            "New Scene Draft",
            height=400,
            placeholder="Paste or type your scene text here...",
            key="screenplay_draft_input",
        )

        col_process, col_verify = st.columns(2)
        with col_process:
            sp_process_clicked = st.button(
                "⚙️ Process Scene",
                use_container_width=True,
                type="primary",
                help="Extract character/prop state from this scene.",
            )
        with col_verify:
            sp_verify_clicked = st.button(
                "🔍 Check Continuity",
                use_container_width=True,
                help="Check this scene draft against established continuity.",
            )

        if sp_process_clicked:
            if not scene_text.strip():
                st.error("Please enter scene text before processing.")
            else:
                try:
                    with st.spinner(f"Processing scene {st.session_state.scene_number}..."):
                        sp_vs = get_vector_store(
                            persist_directory=SCREENPLAY_CHROMA_PATH,
                            collection_name="scenes",
                        )
                        n_chunks = ingest_chapter(
                            scene_text, st.session_state.scene_number, sp_vs,
                        )
                        current_sp = load_screenplay_state(SCREENPLAY_STATE_PATH)
                        updated_sp = extract_scene_state(
                            scene_text, st.session_state.scene_number, current_sp,
                        )
                        save_screenplay_state(updated_sp, SCREENPLAY_STATE_PATH)

                        st.session_state.scene_number += 1
                        st.session_state.screenplay_process_success = True
                        st.session_state.screenplay_verification_result = None

                    st.success(
                        f"✅ Scene {st.session_state.scene_number - 1} processed! "
                        f"({n_chunks} chunks ingested, "
                        f"{len(updated_sp.characters)} character(s), "
                        f"{len(updated_sp.props)} prop(s) tracked)"
                    )
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing scene: {str(e)}")
                    st.exception(e)

        if sp_verify_clicked:
            if not scene_text.strip():
                st.error("Please enter scene text before verifying.")
            else:
                try:
                    with st.spinner("Checking for continuity issues..."):
                        current_sp = load_screenplay_state(SCREENPLAY_STATE_PATH)
                        sp_vs = get_vector_store(
                            persist_directory=SCREENPLAY_CHROMA_PATH,
                            collection_name="scenes",
                        )
                        retrieved = retrieve_relevant_context(
                            query_text=scene_text, vector_store=sp_vs,
                            k=5, exclude_chapter=st.session_state.scene_number,
                        )
                        sp_result = verify_scene(scene_text, current_sp, retrieved)
                        st.session_state.screenplay_verification_result = sp_result
                except Exception as e:
                    st.error(f"❌ Error during verification: {str(e)}")
                    st.exception(e)

        if st.session_state.screenplay_verification_result is not None:
            st.divider()
            render_continuity_flags(
                st.session_state.screenplay_verification_result.flags,
                "✅ **No continuity issues detected.** The scene is consistent with established state.",
            )

        if st.session_state.screenplay_process_success:
            st.session_state.screenplay_process_success = False

    with tab_graph:
        st.subheader("Screenplay Continuity Graph")
        graph_sp = load_screenplay_state(SCREENPLAY_STATE_PATH)

        if not graph_sp.characters and not graph_sp.props:
            st.info("No state tracked yet — process a scene in the **Continuity Checker** tab first.")
        else:
            nodes, edges = build_screenplay_graph_elements(graph_sp)
            agraph(nodes=nodes, edges=edges, config=get_graph_config())

            st.caption(
                "🔵 **Blue dot** = Healthy character  ·  "
                "🔴 **Red dot** = Injured character  ·  "
                "🟩 **Teal box** = Active prop  ·  "
                "**Arrows** = Character-prop associations"
            )

            destroyed = [p for p in graph_sp.props if p.status == "destroyed"]
            if destroyed:
                with st.expander("💥 Destroyed props (not shown in graph)"):
                    for prop in destroyed:
                        st.markdown(
                            f"- **{prop.name}** — last seen: "
                            f"{prop.last_seen_location} (scene {prop.last_seen_scene})"
                        )


# ===================================================================
# STANDUP COMEDY DOMAIN
# ===================================================================

elif domain == "Standup Comedy":

    with st.sidebar:
        st.title("🎤 Set State")
        st.caption("Live state from `storage/standup_state.json`")
        st.divider()

        su_state = load_set_state(STANDUP_STATE_PATH)

        if not su_state.premises:
            st.info("No premises tracked yet. Process a set to begin.")
        else:
            st.markdown("### Premises")
            status_icons = {"open": "🔴", "resolved": "🟢", "callback_used": "🟡"}
            for p in su_state.premises:
                icon = status_icons.get(p.status, "⚪")
                st.markdown(f"{icon} **{p.premise_id}** [{p.category}] — {p.description}")
                if p.callback_segments:
                    st.caption(f"  Callbacks in segments: {p.callback_segments}")
                if p.resolution_note:
                    st.caption(f"  Resolution: {p.resolution_note}")
            st.divider()

            if su_state.punchlines:
                st.markdown("### Punchlines")
                for pl in su_state.punchlines:
                    st.markdown(
                        f"🎯 *\"{pl.text}\"* → {pl.attached_to_premise_id} "
                        f"(seg {pl.segment_delivered})"
                    )
                st.divider()

            if su_state.tags:
                st.markdown("### Tags")
                for tag in su_state.tags:
                    st.markdown(
                        f"🏷️ *\"{tag.text}\"* → {tag.attached_to_premise_id} "
                        f"(seg {tag.segment_introduced})"
                    )
                st.divider()

    st.title("🎤 Standup Comedy Coverage Engine")
    st.markdown(
        "Paste your full standup transcript (with `[SEGMENT]` delimiters) "
        "and click **Process Full Set** to classify all premises, punchlines, "
        "tags, and callbacks in one pass."
    )

    tab_processor, tab_coverage, tab_graph = st.tabs(
        ["🎙️ Set Processor", "📊 Coverage Check", "🕸️ Knowledge Graph"]
    )

    with tab_processor:
        full_set_text = st.text_area(
            "Full Standup Transcript",
            height=400,
            placeholder=(
                "Paste your full standup transcript here...\n\n"
                "Use [SEGMENT] delimiters to separate segments.\n\n"
                "[SEGMENT]\nFirst bit goes here...\n\n"
                "[SEGMENT]\nSecond bit goes here..."
            ),
            key="standup_full_set_input",
        )

        col_process, col_reset = st.columns(2)
        with col_process:
            su_process_clicked = st.button(
                "⚙️ Process Full Set",
                use_container_width=True,
                type="primary",
                help="Split on [SEGMENT], classify each segment sequentially, carry state forward.",
            )
        with col_reset:
            su_reset_clicked = st.button(
                "🗑️ Reset State",
                use_container_width=True,
                help="Clear all standup state and start fresh.",
            )

        if su_reset_clicked:
            save_set_state(SetState(), STANDUP_STATE_PATH)
            st.session_state.standup_process_success = False
            st.session_state.coverage_result = None
            st.session_state.standup_processing_log = []
            st.success("✅ State reset. Ready for a fresh set.")
            st.rerun()

        if su_process_clicked:
            if not full_set_text.strip():
                st.error("Please paste your standup transcript before processing.")
            else:
                SEGMENT_DELIMITER = "[SEGMENT]"
                segments = [s.strip() for s in full_set_text.split(SEGMENT_DELIMITER) if s.strip()]

                if not segments:
                    st.error("No segments found. Make sure your transcript uses `[SEGMENT]` delimiters.")
                else:
                    state = SetState()
                    processing_log = []
                    progress_bar = st.progress(0, text="Starting...")

                    try:
                        for seg_idx, segment_text in enumerate(segments):
                            seg_num = seg_idx + 1
                            progress_bar.progress(
                                seg_idx / len(segments),
                                text=f"Processing segment {seg_num}/{len(segments)}...",
                            )
                            state = extract_segment_state(segment_text, seg_num, state)
                            processing_log.append(
                                f"Segment {seg_num}: "
                                f"{len(state.premises)} premise(s), "
                                f"{len(state.punchlines)} punchline(s), "
                                f"{len(state.tags)} tag(s)"
                            )

                        save_set_state(state, STANDUP_STATE_PATH)
                        progress_bar.progress(1.0, text="Done!")
                        st.session_state.standup_process_success = True
                        st.session_state.coverage_result = None
                        st.session_state.standup_processing_log = processing_log

                        st.success(
                            f"✅ Processed {len(segments)} segment(s)! "
                            f"({len(state.premises)} premises, "
                            f"{len(state.punchlines)} punchlines, "
                            f"{len(state.tags)} tags tracked)"
                        )
                        st.rerun()

                    except Exception as e:
                        progress_bar.empty()
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.exception(e)

        if st.session_state.standup_processing_log:
            with st.expander("📜 Processing Log", expanded=False):
                for entry in st.session_state.standup_processing_log:
                    st.text(entry)

        if st.session_state.standup_process_success:
            st.session_state.standup_process_success = False

    # Coverage tab (deterministic, no LLM call)
    with tab_coverage:
        st.subheader("📊 Coverage Analysis")
        su_state_for_cov = load_set_state(STANDUP_STATE_PATH)

        if not su_state_for_cov.premises:
            st.info("No premises exist yet. Process a full set first to build the set state.")
        else:
            coverage_clicked = st.button(
                "🔍 Check Coverage",
                use_container_width=True,
                type="primary",
                help="Find abandoned setups — deterministic check, no LLM call.",
            )

            if coverage_clicked:
                st.session_state.coverage_result = {
                    "abandoned": find_abandoned_setups(su_state_for_cov),
                    "unpaid": find_premises_without_punchlines(su_state_for_cov),
                    "density": compute_laugh_density(su_state_for_cov),
                }

            if st.session_state.coverage_result is not None:
                abandoned = st.session_state.coverage_result["abandoned"]
                unpaid = st.session_state.coverage_result["unpaid"]
                density = st.session_state.coverage_result["density"]

                st.divider()

                if abandoned:
                    st.warning(f"⚠️ **{len(abandoned)} abandoned setup(s) detected:**")
                    for p in abandoned:
                        st.markdown(
                            f"- 🔴 **{p.premise_id}** [{p.category}]: "
                            f"{p.description} *(introduced in segment {p.introduced_in_segment})*"
                        )
                else:
                    st.success("✅ **No abandoned setups.** All premises have been resolved or called back.")

                if unpaid:
                    st.warning(f"⚠️ **{len(unpaid)} premise(s) without a punchline:**")
                    for p in unpaid:
                        st.markdown(f"- 🎯 **{p.premise_id}**: {p.description} *(setup introduced but never paid off)*")

                st.divider()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Laugh Density", f"{density['overall_density']:.2f}")
                with col2:
                    st.metric("Premises", density["total_premises"])
                with col3:
                    st.metric("Punchlines", density["total_punchlines"])
                with col4:
                    st.metric("Tags", density["total_tags"])

                st.markdown("**Per-premise laugh breakdown:**")
                import pandas as pd
                df_data = []
                for pid, counts in density["per_premise"].items():
                    premise = next((p for p in su_state_for_cov.premises if p.premise_id == pid), None)
                    df_data.append({
                        "Premise ID": pid,
                        "Category": premise.category if premise else "—",
                        "Status": premise.status if premise else "—",
                        "Punchlines": counts["punchlines"],
                        "Tags": counts["tags"],
                        "Total": counts["total"],
                    })
                if df_data:
                    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

    with tab_graph:
        st.subheader("Premise / Punchline / Tag / Callback Graph")
        graph_su = load_set_state(STANDUP_STATE_PATH)

        if not graph_su.premises:
            st.info("No premises tracked yet — process a full set in the **Set Processor** tab first.")
        else:
            nodes, edges = build_standup_graph_elements(graph_su)
            agraph(nodes=nodes, edges=edges, config=get_graph_config())

            st.caption(
                "🔴 **Red dot** = Open premise  ·  "
                "🟢 **Green dot** = Resolved premise  ·  "
                "🔵 **Blue triangle** = Punchline  ·  "
                "💎 **Purple diamond** = Tag  ·  "
                "⬜ **Gray box** = Segment callback marker"
            )

