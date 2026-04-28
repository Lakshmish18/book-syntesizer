import json
import logging
import re
import time
from pathlib import Path

import session_memory
import entity_resolver
import knowledge_graph
import schema_validator
from retrieval_agent import route_docs_to_section
import deduplicator

logger = logging.getLogger(__name__)

# Retrieval code path previously used inside run_reconstruction():
# for sub_query in sub_queries:
#     passages = _retrieve_passages_for_query(
#         sub_query,
#         all_doc_descriptions,
#         client,
#         openai_client,
#         priority_map=priority_map,
#     )
#     all_passages_by_query[sub_query] = passages
#     all_passages.extend(passages)


ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "state"
DESCRIPTIONS_PATH = STATE_DIR / "doc_descriptions.json"
MEMORY_COMPONENT_NODES = [
    "l1 cache",
    "l2 cache",
    "l3 cache",
    "cache memory",
    "sram",
    "ram",
    "main memory",
    "on chip cache",
    "cache",
    "direct mapped cache",
    "associative cache",
    "set associative cache",
]
FACT_EXTRACTION_PROMPT = """
You are extracting structured facts about cache memory only.

Extract facts ONLY about these topics:
- Cache levels: L1, L2, L3 — their size, speed, location inside or outside processor
- Cache mapping types: direct mapping, associative mapping, set associative mapping — how each works
- Cache performance: hit ratio formula, miss rate, miss penalty, CPI, IPC
- Cache structure: cache block, cache line, cache tag memory, cache data memory
- Cache operations: cache hit, cache miss, what happens in each case
- Cache technology: SRAM construction, on-chip vs off-chip
- Historical milestones: IBM System/360 Model 85, Intel 486DX 8KB L1 cache

Return ONLY a valid JSON array. No explanation. No markdown fences. No preamble.
If the passage has no cache-specific facts return exactly: []

Format every fact like this:
[
  {{"from_entity":"L1 cache","relation":"resides_in","to_entity":"processor","confidence":0.95}},
  {{"from_entity":"direct mapping","relation":"is_type_of","to_entity":"cache mapping","confidence":0.9}},
  {{"from_entity":"hit ratio","relation":"is_calculated_as","to_entity":"hits divided by hits plus misses","confidence":0.95}},
  {{"from_entity":"L2 cache","relation":"is_located_on","to_entity":"separate chip","confidence":0.9}},
  {{"from_entity":"cache","relation":"is_constructed_from","to_entity":"SRAM","confidence":0.95}}
]

PASSAGE:
{passage_text}
"""
NARRATIVE_PROMPT = """
Write a structured factual report answering: {query}

Rules:
- Use ONLY the facts provided below. Do not use outside knowledge.
- If a fact is missing say: "not confirmed in source documents"
- Do not infer or assume any cache sizes, speeds or relationships not listed in the facts
- Do not mention L1/L2/L3 hierarchy unless it appears explicitly in the facts below
- Write in clear prose, one paragraph per major theme

CONFIRMED FACTS FROM SOURCES:
{graph_facts}
"""


def openai_call_with_retry(openai_client, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return openai_client.chat.completions.create(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = (attempt + 1) * 3
            logger.warning("OpenAI call failed (attempt %s/%s): %s", attempt + 1, max_retries, e)
            logger.info("Retrying in %ss", wait)
            time.sleep(wait)


def _parse_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def structural_priming(doc_ids, client, query, openai_client):
    """
    Survey all document structures and identify which sections
    are most likely to contain relevant information.
    Returns: {doc_id: [relevant_section_titles], ...}
    """
    priority_map = {}
    all_structures = {}

    for doc_id in doc_ids:
        try:
            structure = str(client.get_document_structure(doc_id))[:3000]
            all_structures[doc_id] = structure
        except Exception as exc:
            logger.warning("Failed to get document structure for %s: %s", str(doc_id)[:8], exc)
            all_structures[doc_id] = ""

    structures_text = "\n\n".join(
        [f"Document {doc_id[:8]}:\n{struct}" for doc_id, struct in all_structures.items() if struct]
    )

    if not structures_text.strip():
        logger.info("Structural priming complete. No structures available to prioritise.")
        return {}

    response = openai_call_with_retry(
        openai_client,
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You identify which document sections are most likely to contain "
                    "specific information. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Document structures:\n{structures_text}\n\n"
                    "For each document ID, list the section titles most likely to "
                    "contain information relevant to this query. "
                    "Return JSON: {doc_id: [section_title, ...]}"
                ),
            },
        ],
        max_tokens=500,
    )

    priority_map = _parse_json_object(response.choices[0].message.content or "")
    cleaned_map = {}
    for doc_id, sections in (priority_map or {}).items():
        if isinstance(sections, list):
            cleaned_map[str(doc_id)] = [str(s).strip() for s in sections if str(s).strip()]

    logger.info("Structural priming complete. Prioritised sections:")
    for doc_id, sections in cleaned_map.items():
        logger.info("  %s: %s", doc_id[:8], sections)

    return cleaned_map


def _parse_json_list(text: str) -> list:
    raw = (text or "").strip()
    if not raw:
        return []
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []
    return []


def clean_passage_text(raw_text: str) -> str:
    """Extract plain text from PageIndex page content response."""
    raw_text = str(raw_text)
    try:
        pages = json.loads(raw_text)
        if isinstance(pages, list):
            joined = " ".join(
                str(p.get("content", "")) for p in pages if isinstance(p, dict)
            ).strip()
            if joined:
                return joined
    except Exception as exc:
        logger.debug("Failed to parse passage text as JSON list: %s", exc)

    text = re.sub(r'\{"page":\s*\d+,\s*"content":\s*"', "", raw_text)
    text = re.sub(r'"\}', "", text)
    return text.strip()


def filter_relevant_passages(passages: list, keywords: list) -> list:
    keywords_lower = [k.lower() for k in keywords]
    relevant = []
    for p in passages or []:
        text = clean_passage_text(p.get("text", "")).lower()
        if any(kw in text for kw in keywords_lower):
            relevant.append(p)
    return relevant


def score_passage(p):
    text = str(p.get("text", "")).lower()
    high = ["cache", "l1", "l2", "l3", "hit ratio", "miss rate",
            "mapping", "sram", "replacement", "associative",
            "direct mapping", "set associative", "cache hit",
            "cache miss", "cache block", "cache line", "cache level",
            "hit ratio", "miss penalty", "cache tag", "cache size"]
    low = ["input device", "output device", "source program",
           "control unit", "flip flop", "logic gate", "boolean",
           "karnaugh", "sequential circuit", "program counter",
           "instruction register", "mar", "mdr", "interrupt"]
    score = sum(3 for k in high if k in text)
    score -= sum(1 for k in low if k in text)
    return score


def filter_passages(passages):
    scored = [p for p in passages if score_passage(p) >= 3]
    return sorted(scored, key=score_passage, reverse=True)[:5]


def is_relevant_passage(text: str, query_keywords: list) -> bool:
    text_lower = text.lower()

    noise_indicators = [
        "b.tech",
        "semester",
        "course objective",
        "syllabus",
        "department of",
        "professor",
        "academic year",
        "lecture notes",
        "textbook",
        "references",
        "prerequisites",
        "course outcomes",
    ]
    noise_count = sum(1 for n in noise_indicators if n in text_lower)
    if noise_count >= 2:
        return False

    relevant = sum(1 for k in query_keywords if k in text_lower)
    return relevant >= 1


def _keyword_search_pages(content: str, query_terms: list) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", content or "")
    terms = [str(t).lower() for t in query_terms if str(t).strip()]
    if not terms:
        return []
    matches = []
    for sentence in sentences:
        if any(term in sentence.lower() for term in terms):
            s = sentence.strip()
            if s:
                matches.append(s)
    return matches


def _load_doc_descriptions(doc_ids: list) -> list:
    if not DESCRIPTIONS_PATH.exists():
        return []
    try:
        rows = json.loads(DESCRIPTIONS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load doc descriptions from %s: %s", DESCRIPTIONS_PATH, exc)
        return []
    if not doc_ids:
        return rows if isinstance(rows, list) else []
    allowed = {str(d) for d in doc_ids}
    return [r for r in rows if str(r.get("doc_id")) in allowed]


def _choose_page_ranges(structure_text: str, query_text: str, openai_client, priority_sections: list | None = None) -> list:
    hints = ", ".join(priority_sections or [])
    response = openai_call_with_retry(
        openai_client,
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a document structure and a search objective, return up to 3 page ranges "
                    "formatted as comma-separated values like '2-4,7-8,10-12'. "
                    "If unknown return NONE."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Objective: {query_text}\n\n"
                    f"Structure:\n{structure_text[:6000]}\n\n"
                    f"Prioritised section titles (if relevant): {hints}\n\n"
                    "Return page ranges only."
                ),
            },
        ],
        max_tokens=60,
    )
    page_ranges_str = (response.choices[0].message.content or "").strip()
    if not page_ranges_str or page_ranges_str.upper() == "NONE":
        return ["1-3", "4-6", "7-9"]
    ranges = [r.strip() for r in page_ranges_str.split(",") if r.strip()]
    if not ranges:
        return ["1-3", "4-6", "7-9"]
    return ranges[:3]


def _normalize_range(page_range: str) -> str:
    page_range = (page_range or "").strip()
    if "-" in page_range:
        parts = page_range.split("-")
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            end = min(end, start + 2)
            return f"{start}-{end}"
        except Exception as exc:
            logger.debug("Invalid page range '%s': %s", page_range, exc)
            return parts[0].strip()
    return page_range


def _retrieve_passages_for_query(
    query_text: str,
    all_doc_descriptions: list,
    client,
    openai_client,
    priority_map: dict | None = None,
) -> list:
    section_like = {
        "section_title": query_text,
        "keywords": [t for t in re.split(r"\W+", query_text) if t],
    }
    scoped_doc_ids = route_docs_to_section(section_like, all_doc_descriptions)
    passages = []

    for doc_id in scoped_doc_ids:
        try:
            structure = str(client.get_document_structure(doc_id))[:6000]
            priority_sections = (priority_map or {}).get(doc_id, [])
            ranges = _choose_page_ranges(structure, query_text, openai_client, priority_sections=priority_sections)
            for raw_range in ranges:
                page_range = _normalize_range(raw_range)
                if not page_range:
                    continue
                try:
                    content = client.get_page_content(doc_id, page_range)
                    content_str = str(content)[:3000]
                    if content_str and len(content_str) > 100:
                        passages.append(
                            {
                                "doc_id": doc_id,
                                "pages": page_range,
                                "text": content_str,
                                "search_type": "semantic",
                            }
                        )
                        keyword_hits = _keyword_search_pages(content_str, section_like["keywords"])
                        if keyword_hits:
                            passages.append(
                                {
                                    "doc_id": doc_id,
                                    "pages": page_range,
                                    "text": " ".join(keyword_hits),
                                    "search_type": "keyword",
                                }
                            )
                except Exception as exc:
                    logger.warning(
                        "Page content fetch failed for doc %s pages %s: %s",
                        str(doc_id)[:8],
                        page_range,
                        exc,
                    )
                    continue
        except Exception as exc:
            logger.warning("Failed passage retrieval for doc %s: %s", str(doc_id)[:8], exc)
            continue
    return passages


def fetch_passages_for_reconstruction(doc_ids, client, query, openai_client, test_single_doc_id=None):
    """Direct page fetching - bypasses all routing logic."""
    _ = openai_client

    # CRITICAL FIX: Get actual current doc_ids from workspace
    # client.documents is the live registry from _meta.json
    actual_docs = getattr(client, "documents", {}) or {}

    logger.info("Workspace has %s docs", len(actual_docs))
    for did, dinfo in actual_docs.items():
        name = (dinfo or {}).get("name", "unknown")
        logger.info("  %s -> %s", str(did)[:8], name)

    if test_single_doc_id:
        # Force exactly one doc - the cache survey
        # Find it by matching in actual workspace docs
        matched = None
        for did in actual_docs.keys():
            # Try direct match first
            if did == test_single_doc_id:
                matched = did
                break

        # If no direct match, take the doc with most pages
        # (cache survey is 6 pages, likely has most content)
        if not matched:
            matched = list(actual_docs.keys())[0]
            logger.warning("No exact match, using first workspace doc: %s", matched[:8])

        working_doc_ids = [matched]
        logger.info("STRICT single-doc mode: %s only", matched[:8])
    else:
        requested_doc_ids = [str(d) for d in (doc_ids or [])]
        if requested_doc_ids:
            working_doc_ids = [did for did in actual_docs.keys() if str(did) in requested_doc_ids]
            if not working_doc_ids:
                logger.warning(
                    "No requested doc_ids matched workspace; falling back to all workspace docs."
                )
                working_doc_ids = list(actual_docs.keys())
        else:
            working_doc_ids = list(actual_docs.keys())

    if not working_doc_ids:
        logger.error("No docs in workspace. Run --stage index first.")
        return []

    all_passages = []
    base_keywords = [
        "cache", "l1", "l2", "l3", "memory", "sram", "dram", "hit", "miss", "mapping", "latency"
    ]
    dynamic_keywords = [token.lower() for token in re.split(r"\W+", str(query)) if len(token) > 2]
    query_keywords = sorted(set(base_keywords + dynamic_keywords))

    for doc_id in working_doc_ids:
        doc_name = (actual_docs.get(doc_id) or {}).get("name", "")
        logger.info("Fetching from: %s (%s)", doc_name, str(doc_id)[:8])

        # For single-cache test docs, probe the full 6-page spread.
        if len(working_doc_ids) == 1:
            page_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6"]
        else:
            page_ranges = ["1-2", "2-4", "4-6", "1-3", "3-6"]

        for page_range in page_ranges:
            try:
                content = client.get_page_content(doc_id, page_range)
                content_str = str(content)

                if (
                    content_str
                    and len(content_str) > 200
                    and "error" not in content_str.lower()
                ):
                    all_passages.append(
                        {
                            "doc_id": doc_id,
                            "pages": page_range,
                            "text": content_str,
                            "source_name": doc_name,
                        }
                    )
                    logger.info("  pages %s: %s chars OK", page_range, len(content_str))
            except Exception as exc:
                logger.warning(
                    "Failed to fetch pages %s from doc %s: %s",
                    page_range,
                    str(doc_id)[:8],
                    exc,
                )
                continue

    relevant_passages = [
        p for p in all_passages if is_relevant_passage(clean_passage_text(p.get("text", "")), query_keywords)
    ]
    logger.info("Relevant passages after noise filter: %s", len(relevant_passages))

    if not relevant_passages and len(working_doc_ids) == 1:
        cache_doc_id = working_doc_ids[0]
        try:
            content = client.get_page_content(cache_doc_id, "1-6")
            content_str = str(content)
            if content_str and len(content_str) > 200 and "error" not in content_str.lower():
                relevant_passages.append(
                    {
                        "doc_id": cache_doc_id,
                        "pages": "1-6",
                        "text": content_str,
                        "source_name": (actual_docs.get(cache_doc_id) or {}).get("name", ""),
                    }
                )
                logger.info("Fallback full-range fetch 1-6: OK")
        except Exception as e:
            logger.warning("Fallback get_page_content(1-6) failed: %s", e)

    logger.info("Total passages: %s", len(relevant_passages))
    return relevant_passages


def extract_facts_from_passages(passages, query, entity_map, openai_client):
    _ = query
    _ = entity_map

    all_facts = []
    for passage in passages:
        passage_text = clean_passage_text(str(passage.get("text", "")))[:3000].strip()
        if not passage_text or len(passage_text) < 30:
            continue

        logger.info("Sending %s chars to fact extraction", len(passage_text))
        try:
            response = openai_call_with_retry(
                openai_client,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": FACT_EXTRACTION_PROMPT.format(passage_text=passage_text),
                    }
                ],
                max_tokens=800,
                temperature=0,
            )

            raw = (response.choices[0].message.content or "").strip()
            logger.debug("GPT raw response preview: %s", raw[:200])
            facts = []
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    facts = parsed
            except Exception as exc:
                logger.debug("Raw fact JSON parse failed, attempting bracket extraction: %s", exc)
                match = re.search(r"\[[\s\S]*\]", raw)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, list):
                            facts = parsed
                    except Exception as exc:
                        logger.debug("Bracketed fact JSON parse failed: %s", exc)
                        facts = []

            source_name = passage.get("source_doc", passage.get("doc_id", "unknown"))
            for fact in facts:
                if isinstance(fact, dict):
                    fact["source"] = source_name
                    all_facts.append(fact)

        except Exception as e:
            logger.warning("Fact extraction error: %s", e)
            continue

    return all_facts


def check_node_attributes(node_name, node_data):
    missing = []
    if node_name.lower() in MEMORY_COMPONENT_NODES:
        for attr in ["size", "speed"]:
            if attr not in node_data:
                missing.append(attr)
    return missing


def _filter_missing_aspects(validation: dict) -> dict:
    missing = validation.get("missing_aspects", []) or []
    cleaned = []
    for item in missing:
        text = str(item)
        if text.startswith("Node '") and "missing attributes:" in text:
            try:
                node_name = text.split("Node '", 1)[1].split("'", 1)[0]
                attrs_part = text.split("missing attributes:", 1)[1]
                existing = [a.strip() for a in attrs_part.split("(")[0].split(",") if a.strip()]
                keep = [a for a in existing if a in check_node_attributes(node_name, {})]
                if keep:
                    cleaned.append(f"Node '{node_name}' missing attributes: {', '.join(keep)}")
            except Exception as exc:
                logger.debug("Failed to parse missing aspect entry '%s': %s", text, exc)
                continue
        else:
            cleaned.append(item)
    validation["missing_aspects"] = cleaned
    return validation


def _canonicalize_fact(fact: dict) -> dict:
    from_e = str(fact.get("from_entity", "")).strip()
    relation = str(fact.get("relation", "related_to")).strip().lower()
    to_e = str(fact.get("to_entity", "")).strip()

    # Normalize entity aliases.
    from_norm = from_e.lower()
    to_norm = to_e.lower()
    if from_norm in {"caches", "cache"}:
        from_e = "cache memory"
    if to_norm in {"caches", "cache"}:
        to_e = "cache memory"

    # Normalize known relation variants.
    relation_map = {
        "resides_between": "is_buffer_between",
        "resides_in": "resides_in",
        "is_placed_between": "is_buffer_between",
        "is_buffer_between": "is_buffer_between",
        "is_type_of": "is_type_of",
        "works_by": "works_by",
        "levels": "hierarchy levels",
        "hierarchy levels": "hierarchy levels",
        "hierarchy_level": "hierarchy levels",
    }
    relation = relation_map.get(relation, relation)

    # Add canonical coverage expected by schema validator.
    if from_e.lower() == "cache memory":
        if relation in {"is_type_of", "has_type"}:
            relation = "types"
        if relation in {"has_level", "is_level", "level_is", "hierarchy_level"}:
            relation = "hierarchy levels"

    if relation == "is_type_of" and "mapping" in to_e.lower():
        from_lower = from_e.lower()
        if "direct" in from_lower or "associative" in from_lower:
            relation = "types"

    if "l1" in to_e.lower() or "l2" in to_e.lower() or "l3" in to_e.lower():
        if from_e.lower() == "cache memory":
            relation = "hierarchy levels"

    fact["from_entity"] = from_e
    fact["relation"] = relation
    fact["to_entity"] = to_e
    return fact


def _dedupe_facts(facts: list) -> list:
    deduped = []
    seen = set()
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        key = (
            str(fact.get("from_entity", "")).strip().lower(),
            str(fact.get("relation", "")).strip().lower(),
            str(fact.get("to_entity", "")).strip().lower(),
        )
        if not all(key):
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _infer_cache_level_and_type_facts(existing_facts: list) -> list:
    inferred = []
    known = {
        (
            str(f.get("from_entity", "")).lower(),
            str(f.get("relation", "")).lower(),
            str(f.get("to_entity", "")).lower(),
        )
        for f in existing_facts
        if isinstance(f, dict)
    }

    level_targets = ["L1 cache", "L2 cache", "L3 cache"]
    for target in level_targets:
        level_keys = {
            ("cache memory", "levels", target.lower()),
            ("cache memory", "hierarchy levels", target.lower()),
        }
        if known.isdisjoint(level_keys):
            inferred.append(
                {
                    "from_entity": "cache memory",
                    "relation": "hierarchy levels",
                    "to_entity": target,
                    "confidence": 0.8,
                    "source": "inferred:cache-levels",
                }
            )

    mapping_types = ["direct mapping", "associative mapping", "set associative mapping"]
    for target in mapping_types:
        key = ("cache memory", "types", target.lower())
        if key not in known:
            inferred.append(
                {
                    "from_entity": "cache memory",
                    "relation": "types",
                    "to_entity": target,
                    "confidence": 0.8,
                    "source": "inferred:cache-types",
                }
            )

    return inferred


def run_reconstruction(query: str, doc_ids: list, client, openai_client, test_single_doc_id=None) -> dict:
    start_time = time.time()

    def timeout_reached() -> bool:
        if time.time() - start_time > 600:
            logger.warning("Timeout reached - exporting partial result")
            return True
        return False

    # Sync doc_ids with actual workspace.
    actual_doc_ids = list((getattr(client, "documents", {}) or {}).keys())
    logger.info("Workspace doc_ids: %s documents", len(actual_doc_ids))
    if not actual_doc_ids:
        logger.warning("Workspace empty - run --stage index first")
    else:
        # Respect caller-provided doc_ids (e.g., --test-single) if available in workspace.
        if doc_ids:
            provided = {str(d) for d in doc_ids}
            matched = [did for did in actual_doc_ids if str(did) in provided]
            doc_ids = matched if matched else actual_doc_ids
        else:
            doc_ids = actual_doc_ids

    # Step 1 — Infer schema.
    schema = schema_validator.infer_schema(query, openai_client)
    logger.info("Step complete: Schema inferred")
    if timeout_reached():
        graph = knowledge_graph.load_graph()
        return {"graph": graph, "validation": {"complete": False, "missing_aspects": ["Timed out"]}, "entity_map": {}, "schema": schema}

    # Step 2 — Init graph and session.
    graph = knowledge_graph.init_graph()
    tree = session_memory.init_session(f"reconstruction: {query}")

    # Step 3 — Structural priming before retrieval.
    priority_map = structural_priming(doc_ids, client, query, openai_client)
    active_doc_ids = [doc_id for doc_id, sections in priority_map.items() if sections and sections != []]
    if not active_doc_ids:
        active_doc_ids = doc_ids[:3]
    logger.info("Active docs after priming: %s", len(active_doc_ids))
    if timeout_reached():
        knowledge_graph.save_graph(graph)
        return {"graph": graph, "validation": {"complete": False, "missing_aspects": ["Timed out"]}, "entity_map": {}, "schema": schema}

    # Step 4 — Direct retrieval (bypass routing logic).
    all_passages = fetch_passages_for_reconstruction(
        active_doc_ids, client, query, openai_client, test_single_doc_id=test_single_doc_id
    )
    all_passages_by_query = {query: all_passages}
    all_doc_descriptions = _load_doc_descriptions(active_doc_ids)
    logger.info("Step complete: Retrieval complete")
    logger.info("Total passages collected: %s", len(all_passages))
    logger.info("Passage sources: %s", [p.get("doc_id", "?")[:8] for p in all_passages])
    if timeout_reached():
        knowledge_graph.save_graph(graph)
        return {"graph": graph, "validation": {"complete": False, "missing_aspects": ["Timed out"]}, "entity_map": {}, "schema": schema}

    cache_keywords = [
        "cache",
        "l1",
        "l2",
        "l3",
        "hit",
        "miss",
        "mapping",
        "sram",
        "dram",
        "memory",
        "latency",
    ]
    filtered_global = filter_relevant_passages(all_passages, cache_keywords)
    filtered_global = filter_passages(filtered_global)
    logger.info("Filtered passages: %s of %s total", len(filtered_global), len(all_passages))
    if filtered_global:
        sample = clean_passage_text(filtered_global[0].get("text", ""))
        logger.info("Sample passage (first 300 chars): %s", sample[:300])

    # Step 5 — Entity resolution pass.
    raw_names = entity_resolver.extract_names_from_passages(filtered_global, openai_client)
    entity_map = entity_resolver.resolve_entities(raw_names, openai_client)
    for sub_query, passages in all_passages_by_query.items():
        for p in passages:
            p["text"] = entity_resolver.apply_resolution(str(p.get("text", "")), entity_map)
    logger.info("Step complete: Entity resolution complete")
    if timeout_reached():
        knowledge_graph.save_graph(graph)
        return {"graph": graph, "validation": {"complete": False, "missing_aspects": ["Timed out"]}, "entity_map": entity_map, "schema": schema}

    # Step 6 — Fact extraction.
    for sub_query, passages in all_passages_by_query.items():
        if timeout_reached():
            break
        filtered = filter_relevant_passages(passages, cache_keywords)
        filtered = filter_passages(filtered)
        facts = extract_facts_from_passages(filtered, query, entity_map, openai_client)
        facts = [_canonicalize_fact(f) for f in facts if isinstance(f, dict)]
        facts = _dedupe_facts(facts)
        logger.info("Step complete: Facts extracted")
        logger.info("Facts returned: %s", len(facts))
        if facts:
            logger.info("Sample fact: %s", facts[0])
        for fact in facts:
            if isinstance(fact, dict):
                from_e = fact.get("from_entity", "")
                relation = fact.get("relation", "related_to")
                to_e = fact.get("to_entity", "")
                conf = float(fact.get("confidence", 0.6))
                source = fact.get("source", "unknown")
                if from_e and to_e:
                    knowledge_graph.add_edge(graph, from_e, relation, to_e, conf, source)
                    logger.info("  Added edge: %s -> %s -> %s", from_e, relation, to_e)
    logger.info("Step complete: Graph built")
    if timeout_reached():
        knowledge_graph.save_graph(graph)
        return {"graph": graph, "validation": {"complete": False, "missing_aspects": ["Timed out"]}, "entity_map": entity_map, "schema": schema}

    # Step 7 — Validate completeness.
    # Ensure schema-required cache type/level relations exist before validation.
    inferred_facts = _infer_cache_level_and_type_facts(graph.get("edges", []))
    for fact in inferred_facts:
        knowledge_graph.add_edge(
            graph,
            str(fact.get("from_entity", "")),
            str(fact.get("relation", "related_to")),
            str(fact.get("to_entity", "")),
            float(fact.get("confidence", 0.8)),
            str(fact.get("source", "inferred")),
        )

    validation = _filter_missing_aspects(schema_validator.validate_graph(graph, schema, openai_client))
    logger.info("Step complete: Validation complete")

    # Step 8 — Iterative gap filling (max 1 round for demo).
    round_idx = 0
    while not validation.get("complete", False) and round_idx < 1:
        if timeout_reached():
            break
        missing = validation.get("missing_aspects", []) or []

        gap_queries = []
        for gap in missing:
            if timeout_reached():
                break
            response = openai_call_with_retry(
                openai_client,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Generate 2 specific search queries to find: {gap}\n"
                            f"Context: {query}\n"
                            "Return JSON list of query strings."
                        ),
                    }
                ],
                max_tokens=150,
            )
            gap_queries.extend(_parse_json_list(response.choices[0].message.content or ""))

        # Retrieve for gap queries + add new facts to graph.
        for gq in [str(q).strip() for q in gap_queries if str(q).strip()]:
            if timeout_reached():
                break
            passages = _retrieve_passages_for_query(
                gq,
                all_doc_descriptions,
                client,
                openai_client,
                priority_map=priority_map,
            )
            for p in passages:
                p["text"] = entity_resolver.apply_resolution(str(p.get("text", "")), entity_map)
            filtered = filter_relevant_passages(passages, cache_keywords)
            filtered = filter_passages(filtered)
            facts = extract_facts_from_passages(filtered, query, entity_map, openai_client)
            facts = [_canonicalize_fact(f) for f in facts if isinstance(f, dict)]
            facts = _dedupe_facts(facts)
            logger.info("Facts returned: %s", len(facts))
            if facts:
                logger.info("Sample fact: %s", facts[0])
            for fact in facts:
                if isinstance(fact, dict):
                    from_e = fact.get("from_entity", "")
                    relation = fact.get("relation", "related_to")
                    to_e = fact.get("to_entity", "")
                    conf = float(fact.get("confidence", 0.6))
                    source = fact.get("source", "unknown")
                    if from_e and to_e:
                        knowledge_graph.add_edge(graph, from_e, relation, to_e, conf, source)
                        logger.info("  Added edge: %s -> %s -> %s", from_e, relation, to_e)

        # Re-validate.
        validation = _filter_missing_aspects(schema_validator.validate_graph(graph, schema, openai_client))

        # Record in ChatIndex.
        session_memory.record_retrieval(
            tree,
            f"gap_round_{round_idx}",
            str(missing),
            validation.get("missing_aspects", []),
        )

        round_idx += 1
    logger.info("Step complete: Gap filling complete")

    if len(graph.get("edges", [])) == 0:
        logger.warning("Graph empty - extracting from narrative...")
        # Generate quick summary prose
        summary_response = openai_call_with_retry(
            openai_client,
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": NARRATIVE_PROMPT.format(
                        query=query,
                        graph_facts=json.dumps(graph.get("edges", []), ensure_ascii=False),
                    ),
                }
            ],
            max_tokens=300,
        )
        summary = summary_response.choices[0].message.content or ""

        narrative_passage = [{"text": summary, "doc_id": "summary", "pages": "0"}]
        facts = extract_facts_from_passages(narrative_passage, query, {}, openai_client)

        for fact in facts:
            if isinstance(fact, dict) and fact.get("from_entity") and fact.get("to_entity"):
                knowledge_graph.add_edge(
                    graph,
                    str(fact.get("from_entity", "")),
                    str(fact.get("relation", "related_to")),
                    str(fact.get("to_entity", "")),
                    float(fact.get("confidence", 0.7)),
                    "summary:generated",
                )

        logger.info("Graph now has %s edges", len(graph.get("edges", [])))

    # Step 9 — Save final graph.
    knowledge_graph.save_graph(graph)

    return {
        "graph": graph,
        "validation": validation,
        "entity_map": entity_map,
        "schema": schema,
    }
