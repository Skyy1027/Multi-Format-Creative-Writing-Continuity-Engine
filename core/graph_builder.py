"""Graph builder: converts domain states into streamlit-agraph nodes and edges."""

from typing import Dict, List, Tuple
from streamlit_agraph import Config, Edge, Node

from core.schema import LoreState, RelationshipSet
from core.screenplay_schema import ScreenplayState
from core.standup_schema import SetState

# --- Node & Edge Styling ---
CHARACTER_COLOR = "#4361EE"
CHARACTER_COLOR_DECEASED = "#9E9E9E"
CHARACTER_SIZE = 25
CHARACTER_SHAPE = "dot"

LOCATION_COLOR = "#F72585"
LOCATION_SIZE = 20
LOCATION_SHAPE = "box"

DEFAULT_COLOR = "#7209B7"
DEFAULT_SIZE = 18
DEFAULT_SHAPE = "dot"

EDGE_COLOR = "#555555"

LOCATION_RELATION_TYPES = {"located in", "located at", "traveling to", "arrived at"}

SCREENPLAY_CHARACTER_COLOR = "#4361EE"
SCREENPLAY_CHARACTER_INJURED = "#FF6B6B"
SCREENPLAY_PROP_COLOR = "#2EC4B6"
SCREENPLAY_PROP_SIZE = 18
SCREENPLAY_EDGE_COLOR = "#555555"

PREMISE_COLOR_OPEN = "#FF4444"
PREMISE_COLOR_RESOLVED = "#4CAF50"
PUNCHLINE_COLOR = "#2196F3"
TAG_COLOR = "#9C27B0"
SEGMENT_MARKER_COLOR = "#607D8B"

PREMISE_BASE_SIZE = 22
PREMISE_SIZE_PER_TAG = 4
PUNCHLINE_SIZE = 18
TAG_SIZE = 15
SEGMENT_MARKER_SIZE = 12

STANDUP_EDGE_COLOR = "#666666"
CALLBACK_EDGE_COLOR = "#FF9800"


def _make_location_node(name: str) -> Node:
    return Node(id=name, label=name, size=LOCATION_SIZE, color=LOCATION_COLOR, shape=LOCATION_SHAPE)


def _make_default_node(name: str) -> Node:
    return Node(id=name, label=name, size=DEFAULT_SIZE, color=DEFAULT_COLOR, shape=DEFAULT_SHAPE)


# --- Narrative Graph ---

def build_graph_elements(
    lore_state: LoreState,
    relationship_set: RelationshipSet,
) -> Tuple[List[Node], List[Edge]]:
    """Convert LoreState and active relationships into agraph nodes and edges."""
    nodes_dict: Dict[str, Node] = {}
    edges: List[Edge] = []

    for char in lore_state.characters:
        is_deceased = "deceased" in char.physical_status.lower()
        color = CHARACTER_COLOR_DECEASED if is_deceased else CHARACTER_COLOR
        nodes_dict[char.name] = Node(
            id=char.name,
            label=char.name,
            size=CHARACTER_SIZE,
            color=color,
            shape=CHARACTER_SHAPE,
        )

    location_names = {
        rel.target for rel in relationship_set.relationships
        if rel.relation_type.lower() in LOCATION_RELATION_TYPES
    }

    # Only active relationships are shown as live edges
    for rel in relationship_set.relationships:
        if rel.status != "active":
            continue

        if rel.source not in nodes_dict:
            nodes_dict[rel.source] = (
                _make_location_node(rel.source)
                if rel.source in location_names
                else _make_default_node(rel.source)
            )

        if rel.target not in nodes_dict:
            nodes_dict[rel.target] = (
                _make_location_node(rel.target)
                if rel.target in location_names or rel.relation_type.lower() in LOCATION_RELATION_TYPES
                else _make_default_node(rel.target)
            )

        edges.append(Edge(
            source=rel.source,
            target=rel.target,
            label=rel.relation_type,
            color=EDGE_COLOR,
        ))

    return list(nodes_dict.values()), edges


# --- Screenplay Graph ---

def build_screenplay_graph_elements(
    screenplay_state: ScreenplayState,
) -> Tuple[List[Node], List[Edge]]:
    """Convert screenplay character and active prop states into nodes and edges."""
    nodes_dict: Dict[str, Node] = {}
    edges: List[Edge] = []

    for char in screenplay_state.characters:
        is_injured = "injured" in char.physical_status.lower()
        color = SCREENPLAY_CHARACTER_INJURED if is_injured else SCREENPLAY_CHARACTER_COLOR
        nodes_dict[char.name] = Node(
            id=char.name,
            label=f"{char.name}\n({char.current_location})",
            size=CHARACTER_SIZE,
            color=color,
            shape="dot",
        )

    active_props = [
        p for p in screenplay_state.props
        if p.status in ("in_use", "offscreen")
    ]
    for prop in active_props:
        nodes_dict[prop.name] = Node(
            id=prop.name,
            label=f"🎬 {prop.name}\n[{prop.status}]",
            size=SCREENPLAY_PROP_SIZE,
            color=SCREENPLAY_PROP_COLOR,
            shape="box",
        )

    # Connect props to characters based on co-location
    for prop in active_props:
        for char in screenplay_state.characters:
            prop_loc_lower = prop.last_seen_location.lower()
            char_name_lower = char.name.lower()
            char_loc_lower = char.current_location.lower()

            if char_name_lower in prop_loc_lower or prop_loc_lower == char_loc_lower:
                label = "holding" if prop.status == "in_use" else "last had"
                edges.append(Edge(
                    source=char.name,
                    target=prop.name,
                    label=label,
                    color=SCREENPLAY_EDGE_COLOR,
                ))

    return list(nodes_dict.values()), edges


# --- Standup Graph ---

def build_standup_graph_elements(
    set_state: SetState,
) -> Tuple[List[Node], List[Edge]]:
    """Convert standup comedy set state into premises, punchlines, tags, and callback edges."""
    nodes_dict: Dict[str, Node] = {}
    edges: List[Edge] = []

    tag_counts: Dict[str, int] = {}
    for tag in set_state.tags:
        tag_counts[tag.attached_to_premise_id] = tag_counts.get(tag.attached_to_premise_id, 0) + 1

    # Premise nodes (open=red, resolved/callback=green)
    for premise in set_state.premises:
        color = PREMISE_COLOR_OPEN if premise.status == "open" else PREMISE_COLOR_RESOLVED
        n_tags = tag_counts.get(premise.premise_id, 0)
        size = PREMISE_BASE_SIZE + (n_tags * PREMISE_SIZE_PER_TAG)

        nodes_dict[premise.premise_id] = Node(
            id=premise.premise_id,
            label=f"{premise.premise_id}\n[{premise.status}] (seg {premise.introduced_in_segment})",
            size=size,
            color=color,
            shape="dot",
        )

    # Punchlines
    for i, pl in enumerate(set_state.punchlines):
        pl_id = pl.punchline_id or f"pl_{i}_{pl.attached_to_premise_id}"
        nodes_dict[pl_id] = Node(
            id=pl_id,
            label=f"Punchline (seg {pl.segment_delivered})",
            size=PUNCHLINE_SIZE,
            color=PUNCHLINE_COLOR,
            shape="triangle",
        )
        edges.append(Edge(
            source=pl.attached_to_premise_id,
            target=pl_id,
            label="resolved by",
            color=STANDUP_EDGE_COLOR,
        ))

    # Tags
    for i, tag in enumerate(set_state.tags):
        tag_id = f"tag_{i}_{tag.attached_to_premise_id}"
        nodes_dict[tag_id] = Node(
            id=tag_id,
            label=f"Tag (seg {tag.segment_introduced})",
            size=TAG_SIZE,
            color=TAG_COLOR,
            shape="diamond",
        )
        edges.append(Edge(
            source=tag_id,
            target=tag.attached_to_premise_id,
            label="tag",
            color=STANDUP_EDGE_COLOR,
        ))

    # Callbacks
    segment_markers_created = set()
    for premise in set_state.premises:
        for seg_num in premise.callback_segments:
            marker_id = f"Segment {seg_num}"
            if marker_id not in segment_markers_created:
                nodes_dict[marker_id] = Node(
                    id=marker_id,
                    label=f"Seg {seg_num}",
                    size=SEGMENT_MARKER_SIZE,
                    color=SEGMENT_MARKER_COLOR,
                    shape="box",
                )
                segment_markers_created.add(marker_id)

            edges.append(Edge(
                source=marker_id,
                target=premise.premise_id,
                label="callback",
                color=CALLBACK_EDGE_COLOR,
            ))

    return list(nodes_dict.values()), edges


# --- Graph Configuration ---

def get_graph_config() -> Config:
    """Return tuned barnesHut physics config for interactive display."""
    config = Config(
        directed=True,
        physics=True,
        hierarchical=False,
        height=900,
        width=1800,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": True},
    )

    # Set barnesHut parameters on the underlying vis.js dictionary
    config.physics = {
        "enabled": True,
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -30000,
            "centralGravity": 0.1,
            "springLength": 300,
            "springConstant": 0.01,
            "damping": 0.09,
        },
        "minVelocity": 0.75,
        "stabilization": {
            "enabled": True,
            "fit": True,
        },
    }
    return config

