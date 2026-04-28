import json
import re
from collections import Counter
from pathlib import Path

import jellyfish


ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "state"
ENTITY_MAP_PATH = STATE_DIR / "entity_map.json"
OUTLINE_PATH = STATE_DIR / "canonical_outline.json"


def _parse_json_object(text: str):
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _parse_json_list(text: str):
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _load_topic(default: str = "the source corpus") -> str:
    if not OUTLINE_PATH.exists():
        return default
    try:
        payload = json.loads(OUTLINE_PATH.read_text(encoding="utf-8"))
        return payload.get("topic", default)
    except Exception:
        return default


def _cluster_by_similarity(raw_names: list, threshold: float = 0.85):
    names = [str(n).strip() for n in raw_names if str(n).strip()]
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            score = jellyfish.jaro_winkler_similarity(names[i], names[j])
            if score > threshold:
                union(i, j)

    clusters = {}
    for idx, name in enumerate(names):
        root = find(idx)
        clusters.setdefault(root, []).append(name)
    return list(clusters.values())


def resolve_entities(raw_names: list, openai_client) -> dict:
    """
    Given raw name strings extracted from documents, cluster them and
    return a mapping of {raw_name: canonical_name}.
    """
    topic = _load_topic()
    clusters = _cluster_by_similarity(raw_names, threshold=0.85)
    resolution_map = {}

    for cluster in clusters:
        unique_cluster = list(dict.fromkeys([n.strip() for n in cluster if n.strip()]))
        if not unique_cluster:
            continue

        # Singletons resolve to themselves.
        if len(unique_cluster) == 1:
            resolution_map[unique_cluster[0]] = unique_cluster[0]
            continue

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an entity resolution assistant working with names "
                        "from ancient texts. Names often appear in variant spellings "
                        "across different translations. Decide if these names all refer "
                        "to the same person or entity. Return JSON: "
                        "{'same_entity': true/false, 'canonical_name': '...', "
                        "'confidence': 0.0-1.0}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Do these names all refer to the same person?\n"
                        f"Names: {unique_cluster}\n"
                        f"Context: These names appear in documents about {topic}."
                    ),
                },
            ],
        )

        parsed = _parse_json_object(response.choices[0].message.content or "")
        same_entity = bool(parsed.get("same_entity")) if isinstance(parsed, dict) else False
        llm_canonical = parsed.get("canonical_name") if isinstance(parsed, dict) else None

        if same_entity:
            canonical = (str(llm_canonical).strip() if llm_canonical else "")
            if not canonical:
                canonical = Counter(unique_cluster).most_common(1)[0][0]
            for name in unique_cluster:
                resolution_map[name] = canonical
        else:
            for name in unique_cluster:
                resolution_map[name] = name

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ENTITY_MAP_PATH.write_text(
        json.dumps(resolution_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return resolution_map


def extract_names_from_passages(passages: list, openai_client) -> list:
    """
    For retrieved passages, extract person/place/entity names with GPT-4o-mini.
    Returns a deduplicated list of extracted names.
    """
    all_names = set()

    for passage in passages or []:
        text = passage.get("text", "") if isinstance(passage, dict) else str(passage)
        text = (text or "").strip()
        if not text:
            continue

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract all person names, place names, and entity names from this text.\n"
                        "Return a JSON list of strings only.\n\n"
                        f"Text:\n{text[:4000]}"
                    ),
                }
            ],
        )

        names = _parse_json_list(response.choices[0].message.content or "")
        for name in names:
            candidate = str(name).strip()
            if candidate:
                all_names.add(candidate)

    return sorted(all_names)


def apply_resolution(text: str, entity_map: dict) -> str:
    """
    Replace all variant names in text with canonical forms.
    Case-sensitive replacement using word boundaries.
    """
    resolved = text or ""
    if not entity_map:
        return resolved

    # Replace longer keys first to avoid partial overlap conflicts.
    for raw_name in sorted(entity_map.keys(), key=len, reverse=True):
        canonical = entity_map.get(raw_name, raw_name)
        if not raw_name:
            continue
        pattern = r"\b" + re.escape(raw_name) + r"\b"
        resolved = re.sub(pattern, canonical, resolved)
    return resolved
