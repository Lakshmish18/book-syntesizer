import json
from pathlib import Path
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "state"
GRAPH_PATH = STATE_DIR / "knowledge_graph.json"
UNIQUE_RELATIONS = {"is", "equals", "defined_as", "type_is", "version_is", "also_known_as"}


def save_graph(graph) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")


def init_graph() -> dict:
    graph = {"nodes": {}, "edges": []}
    save_graph(graph)
    return graph


def load_graph() -> dict:
    if not GRAPH_PATH.exists():
        return {"nodes": {}, "edges": []}
    try:
        data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"nodes": {}, "edges": []}
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])
        if not isinstance(nodes, dict):
            nodes = {}
        if not isinstance(edges, list):
            edges = []
        return {"nodes": nodes, "edges": edges}
    except Exception:
        return {"nodes": {}, "edges": []}


def _find_node_id_by_canonical_name(graph, name):
    canonical = (name or "").strip()
    for node_id, node in graph.get("nodes", {}).items():
        if str(node.get("canonical_name", "")).strip() == canonical:
            return node_id
    return None


def get_node_by_name(graph, name) -> dict | None:
    node_id = _find_node_id_by_canonical_name(graph, name)
    if node_id is None:
        return None
    return graph.get("nodes", {}).get(node_id)


def add_node(graph, name, node_type, attributes={}, source="") -> str:
    existing = _find_node_id_by_canonical_name(graph, name)
    if existing is not None:
        node = graph["nodes"][existing]
        if source:
            node.setdefault("sources", [])
            if source not in node["sources"]:
                node["sources"].append(source)
        save_graph(graph)
        return existing

    node_id = str(uuid4())
    srcs = [source] if source else []
    graph.setdefault("nodes", {})
    graph["nodes"][node_id] = {
        "canonical_name": (name or "").strip(),
        "type": node_type,
        "attributes": dict(attributes or {}),
        "sources": srcs,
    }
    save_graph(graph)
    return node_id


def _infer_type(name: str) -> str:
    # Default to concept for unknown entities.
    _ = name
    return "concept"


def calculate_confidence(base_confidence, source_count, has_contradiction, source_type):
    """
    3-signal confidence:
    - source_count: more sources = higher confidence
    - has_contradiction: contradicted = lower confidence
    - source_type: 'direct' vs 'inferred'
    """
    count_boost = min(0.2, (max(1, int(source_count)) - 1) * 0.1)
    contradiction_penalty = 0.3 if has_contradiction else 0.0
    type_modifier = 0.0 if str(source_type).lower() == "direct" else -0.1
    final = float(base_confidence) + count_boost - contradiction_penalty + type_modifier
    return max(0.1, min(1.0, final))


def add_edge(graph, from_name, relation, to_name, confidence, source) -> None:
    from_id = add_node(graph, from_name, _infer_type(from_name), source=source)
    to_id = add_node(graph, to_name, _infer_type(to_name), source=source)

    graph.setdefault("edges", [])
    conf = float(confidence) if confidence is not None else 0.0
    conf = max(0.0, min(1.0, conf))

    # Existing same edge: accumulate confidence and source.
    for edge in graph["edges"]:
        if (
            edge.get("from_node") == from_id
            and edge.get("relation") == relation
            and edge.get("to_node") == to_id
        ):
            edge.setdefault("sources", [])
            projected_source_count = len(edge["sources"]) + (1 if source and source not in edge["sources"] else 0)
            edge["confidence"] = calculate_confidence(
                edge.get("confidence", conf),
                projected_source_count,
                len(edge.get("conflicting_sources", []) or []) > 0,
                "direct",
            )
            if source and source not in edge["sources"]:
                edge["sources"].append(source)
            save_graph(graph)
            return

    # Conflict: only for unique relation semantics.
    contradiction_found = False
    for edge in graph["edges"]:
        relation_l = str(relation).lower()
        is_conflict = (
            relation_l in UNIQUE_RELATIONS
            and edge.get("relation") == relation
            and edge.get("from_node") == from_id
            and edge.get("to_node") != to_id
        )
        if not is_conflict:
            continue

        contradiction_found = True
        edge.setdefault("conflicting_sources", [])
        if source and source not in edge["conflicting_sources"]:
            edge["conflicting_sources"].append(source)
        edge["confidence"] = calculate_confidence(
            edge.get("confidence", conf),
            len(edge.get("sources", []) or []),
            True,
            "direct",
        )

    source_type = "inferred" if str(relation).lower() == "inferred" else "direct"
    initial_confidence = calculate_confidence(conf, 1, contradiction_found, source_type)
    new_edge = {
        "from_node": from_id,
        "relation": relation,
        "to_node": to_id,
        "confidence": initial_confidence,
        "sources": [source] if source else [],
        "conflicting_sources": [source] if contradiction_found and source else [],
    }
    graph["edges"].append(new_edge)
    save_graph(graph)


def get_edges_from(graph, node_id) -> list:
    return [edge for edge in graph.get("edges", []) if edge.get("from_node") == node_id]


def get_edges_to(graph, node_id) -> list:
    return [edge for edge in graph.get("edges", []) if edge.get("to_node") == node_id]


def merge_graphs(graph1, graph2) -> dict:
    merged = {"nodes": {}, "edges": []}

    def add_or_merge_node(node_id, node):
        canonical = str(node.get("canonical_name", "")).strip()
        existing_id = _find_node_id_by_canonical_name(merged, canonical)
        if existing_id is None:
            merged["nodes"][node_id] = {
                "canonical_name": canonical,
                "type": node.get("type", "concept"),
                "attributes": dict(node.get("attributes", {})),
                "sources": list(dict.fromkeys(node.get("sources", []))),
            }
            return node_id
        existing = merged["nodes"][existing_id]
        existing["sources"] = list(dict.fromkeys(existing.get("sources", []) + node.get("sources", [])))
        attrs = dict(existing.get("attributes", {}))
        attrs.update(node.get("attributes", {}))
        existing["attributes"] = attrs
        return existing_id

    id_map = {}
    for graph in (graph1 or {"nodes": {}, "edges": []}, graph2 or {"nodes": {}, "edges": []}):
        for old_id, node in graph.get("nodes", {}).items():
            mapped = add_or_merge_node(old_id, node)
            id_map[(id(graph), old_id)] = mapped

    def edge_key(edge):
        return (edge.get("from_node"), edge.get("relation"), edge.get("to_node"))

    edge_index = {}
    for graph in (graph1 or {"nodes": {}, "edges": []}, graph2 or {"nodes": {}, "edges": []}):
        for edge in graph.get("edges", []):
            from_m = id_map.get((id(graph), edge.get("from_node")), edge.get("from_node"))
            to_m = id_map.get((id(graph), edge.get("to_node")), edge.get("to_node"))
            candidate = {
                "from_node": from_m,
                "relation": edge.get("relation"),
                "to_node": to_m,
                "confidence": float(edge.get("confidence", 0.0)),
                "sources": list(edge.get("sources", [])),
                "conflicting_sources": list(edge.get("conflicting_sources", [])),
            }
            key = edge_key(candidate)
            if key in edge_index:
                existing = edge_index[key]
                existing["confidence"] = max(float(existing.get("confidence", 0.0)), candidate["confidence"])
                existing["sources"] = list(dict.fromkeys(existing.get("sources", []) + candidate["sources"]))
                existing["conflicting_sources"] = list(
                    dict.fromkeys(existing.get("conflicting_sources", []) + candidate["conflicting_sources"])
                )
            else:
                merged["edges"].append(candidate)
                edge_index[key] = candidate

    return merged
