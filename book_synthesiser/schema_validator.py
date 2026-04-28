import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "state"
SCHEMA_PATH = STATE_DIR / "reconstruction_schema.json"
VALIDATION_PATH = STATE_DIR / "validation_result.json"


def _save_json(path: Path, payload: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        snippet = raw[start : end + 1]
        try:
            value = json.loads(snippet)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def infer_schema(query: str, openai_client) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract the core subject from a reconstruction query and define "
                    "a completion schema. The subject is the THING being reconstructed, "
                    "not the action. For 'Extract all RAM specifications' the subject is "
                    "'RAM specifications'. For 'Build the Dasharatha family tree' the "
                    "subject is 'Dasharatha family tree'. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    "First identify the core subject being reconstructed (not the action verb).\n"
                    "Then define what a complete answer looks like.\n\n"
                    "Return this JSON:\n"
                    "{\n"
                    "  'target_entity': '<core subject, 2-4 words>',\n"
                    "  'entity_type': 'person|technology|concept|event|relationship',\n"
                    "  'required_relations': ['<specific relation types needed>'],\n"
                    "  'required_attributes': ['<specific attributes needed>'],\n"
                    "  'minimum_nodes': <int>,\n"
                    "  'completeness_questions': [\n"
                    "    '<specific yes/no question about completeness>',\n"
                    "    '<another specific question>'\n"
                    "  ],\n"
                    "  'search_queries': [\n"
                    "    '<specific search string to find this information>',\n"
                    "    '<another search string>'\n"
                    "  ]\n"
                    "}"
                ),
            },
        ],
    )

    schema = _parse_json_object(response.choices[0].message.content or "")
    if not schema:
        schema = {
            "target_entity": " ".join(str(query).split()[-3:]).strip() or "target subject",
            "entity_type": "concept",
            "required_relations": [],
            "required_attributes": [],
            "minimum_nodes": 1,
            "completeness_questions": [],
            "search_queries": [],
        }

    _save_json(SCHEMA_PATH, schema)
    return schema


def _find_node_id_by_canonical_name(graph: dict, canonical_name: str):
    for node_id, node in (graph.get("nodes") or {}).items():
        if str(node.get("canonical_name", "")).strip() == str(canonical_name).strip():
            return node_id
    return None


def _summarize_graph(graph: dict) -> str:
    nodes = graph.get("nodes", {}) or {}
    edges = graph.get("edges", []) or []

    node_lines = []
    for node_id, node in nodes.items():
        node_lines.append(
            f"- {node.get('canonical_name', node_id)} "
            f"(type={node.get('type', 'unknown')}, attributes={node.get('attributes', {})})"
        )

    edge_lines = []
    for edge in edges:
        from_node = nodes.get(edge.get("from_node", ""), {}).get("canonical_name", edge.get("from_node", "unknown"))
        to_node = nodes.get(edge.get("to_node", ""), {}).get("canonical_name", edge.get("to_node", "unknown"))
        edge_lines.append(
            f"- {from_node} --{edge.get('relation', 'related_to')}--> {to_node} "
            f"(confidence={edge.get('confidence', 0.0)})"
        )

    return (
        "Nodes found:\n"
        + ("\n".join(node_lines) if node_lines else "- none")
        + "\n\nRelations found:\n"
        + ("\n".join(edge_lines) if edge_lines else "- none")
    )


def _retrieval_has_content_for_target(target_entity: str) -> bool:
    retrieval_path = STATE_DIR / "retrieval_results.json"
    if not retrieval_path.exists():
        return False
    try:
        rows = json.loads(retrieval_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(rows, list):
        return False

    target_terms = [t for t in str(target_entity).lower().split() if t]
    for section in rows:
        passages = section.get("retrieved_passages", []) or []
        if not passages:
            continue
        section_title = str(section.get("section_title", "")).lower()
        if target_terms and any(term in section_title for term in target_terms):
            return True
        for passage in passages:
            text = str(passage.get("text", "")).lower()
            if text and (not target_terms or any(term in text for term in target_terms)):
                return True
    return False


def _retrieval_has_any_passages() -> bool:
    retrieval_path = STATE_DIR / "retrieval_results.json"
    if not retrieval_path.exists():
        return False
    try:
        rows = json.loads(retrieval_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(rows, list):
        return False
    for section in rows:
        if (section.get("retrieved_passages", []) or []):
            return True
    return False


def validate_graph(graph, schema, openai_client) -> dict:
    nodes = graph.get("nodes", {}) or {}
    edges = graph.get("edges", []) or []

    target_entity = schema.get("target_entity", "")
    required_relations = schema.get("required_relations", []) or []
    required_attributes = schema.get("required_attributes", []) or []
    minimum_nodes = int(schema.get("minimum_nodes", 0) or 0)
    completeness_questions = schema.get("completeness_questions", []) or []

    missing_aspects = []
    conflicts = []
    score_checks = []

    # Step 1 — Structural checks
    retrieval_has_any = _retrieval_has_any_passages()
    type_counts = {}
    for node in nodes.values():
        node_type = str(node.get("type", "unknown"))
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    if len(nodes) < minimum_nodes:
        missing_aspects.append(
            f"Minimum node count not met: {len(nodes)} < {minimum_nodes} "
            "(Not confirmed in graph (but may appear in narrative))"
        )
        if len(nodes) == 0 and retrieval_has_any:
            score_checks.append(0.5)
        else:
            score_checks.append(0.0)
    else:
        score_checks.append(1.0)

    target_node_id = _find_node_id_by_canonical_name(graph, target_entity)
    target_has_retrieval_content = _retrieval_has_content_for_target(target_entity)
    if target_entity and not target_node_id:
        if not target_has_retrieval_content:
            missing_aspects.append(
                f"Target entity not found: {target_entity} "
                "(Not confirmed in graph (but may appear in narrative))"
            )
            score_checks.append(0.0)
        else:
            score_checks.append(0.5)
    elif target_node_id:
        relations_from_target = {str(e.get("relation", "")) for e in edges if e.get("from_node") == target_node_id}
        for rel in required_relations:
            if rel not in relations_from_target:
                missing_aspects.append(
                    f"Missing required relation for target '{target_entity}': {rel} "
                    "(Not confirmed in graph (but may appear in narrative))"
                )
                score_checks.append(0.0)
            else:
                score_checks.append(1.0)

    for node_id, node in nodes.items():
        attrs = node.get("attributes", {}) or {}
        missing_for_node = [attr for attr in required_attributes if attr not in attrs]
        if missing_for_node:
            missing_aspects.append(
                f"Node '{node.get('canonical_name', node_id)}' missing attributes: {', '.join(missing_for_node)} "
                "(Not confirmed in graph (but may appear in narrative))"
            )
            score_checks.append(0.0)
        else:
            if required_attributes:
                score_checks.append(1.0)

    for edge in edges:
        conflicting_sources = edge.get("conflicting_sources", []) or []
        if conflicting_sources:
            conflicts.append(
                f"Conflict on relation {edge.get('from_node')} --{edge.get('relation')}--> {edge.get('to_node')}"
            )
            score_checks.append(0.0)

    # Step 2 — LLM completeness check
    graph_summary = _summarize_graph(graph)
    llm_missing = []
    for question in completeness_questions:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Given what we have found, {question}? "
                        "Answer: complete / partial / missing\n\n"
                        f"We have found:\n{graph_summary}"
                    ),
                }
            ],
            max_tokens=10,
        )
        answer = (response.choices[0].message.content or "").strip().lower()
        if "complete" in answer and "partial" not in answer and "missing" not in answer:
            score_checks.append(1.0)
        elif "partial" in answer:
            llm_missing.append(f"{question} -> partial")
            score_checks.append(0.5)
        else:
            llm_missing.append(f"{question} -> Not confirmed in graph (but may appear in narrative)")
            score_checks.append(0.0)

    missing_aspects.extend(llm_missing)

    completeness_score = 0.0
    if score_checks:
        completeness_score = sum(score_checks) / len(score_checks)

    validation_result = {
        "completeness_score": round(float(completeness_score), 4),
        "missing_aspects": missing_aspects,
        "conflicts": conflicts,
        "complete": bool(completeness_score >= 0.8 and not conflicts and not missing_aspects),
        "node_type_counts": type_counts,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }

    _save_json(VALIDATION_PATH, validation_result)
    return validation_result
