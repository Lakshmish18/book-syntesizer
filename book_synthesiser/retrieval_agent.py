import json
import os
import re
from pathlib import Path

import openai
from agents import set_tracing_disabled
from dotenv import load_dotenv
from openai import OpenAI
from pageindex.client import PageIndexClient
from tqdm import tqdm

import session_memory

set_tracing_disabled(True)


ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "state"
OUTLINE_PATH = STATE_DIR / "canonical_outline.json"
AGENT_STATE_PATH = STATE_DIR / "agent_state.json"
RESULTS_PATH = STATE_DIR / "retrieval_results.json"
DESCRIPTIONS_PATH = STATE_DIR / "doc_descriptions.json"
REGISTRY_PATH = STATE_DIR / "doc_registry.json"


def _load_json(path: Path, default=None):
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _flatten_sections(outline):
    sections = []
    for chapter in outline.get("chapters", []):
        for section in chapter.get("sections", []):
            sections.append(section)
    return sections


def _extract_final_text(run_result):
    for attr in ["final_output", "output_text", "last_output", "text"]:
        value = getattr(run_result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(run_result)


def _extract_json_object(text):
    # Strip markdown fences if present
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"^```\s*", "", text.strip())
    text = re.sub(r"```$", "", text.strip())

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first complete JSON object using brace matching
    brace_count = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    brace_count = 0
    return None


def _parse_agent_output(raw_text):
    parsed = _extract_json_object(raw_text)
    if parsed is not None:
        return parsed
    return {
        "retrieved_passages": [{"doc_id": "unknown", "pages": "unknown", "text": raw_text}],
        "answered_aspects": [],
        "gaps": ["Agent output was not valid JSON; manual review needed"],
    }


def _summarize_found(retrieved_passages, answered_aspects):
    passages_count = len(retrieved_passages)
    aspects = ", ".join(answered_aspects) if answered_aspects else "no explicit answered_aspects provided"
    return f"Retrieved {passages_count} passages. Covered aspects: {aspects}."


def keyword_search_pages(content: str, keywords: list) -> list:
    """
    Given raw page content and a list of keywords,
    return sentences that contain at least one keyword.
    Case-insensitive. Returns list of matching sentence strings.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", content)
    keywords_lower = [str(k).lower() for k in keywords]
    matches = []
    for sentence in sentences:
        if any(kw in sentence.lower() for kw in keywords_lower):
            matches.append(sentence.strip())
    return matches


def retrieve_for_section(section, scoped_doc_ids, client, tree) -> dict:
    section_title = section["section_title"]
    keywords = section.get("keywords", [])
    retrieved_passages = []
    answered_aspects = []
    gaps = []

    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for doc_id in scoped_doc_ids:
        try:
            # Get document metadata
            doc_info = client.get_document(doc_id)

            # Get page count from doc_info
            doc_str = str(doc_info)
            page_count = 10  # default fallback
            try:
                match = re.search(r'page_count["\s:]+(\d+)', doc_str)
                if match:
                    page_count = int(match.group(1))
            except Exception:
                pass

            # Get structure with a wider window so the navigator sees deeper nodes.
            structure = str(client.get_document_structure(doc_id))[:6000]

            # Ask GPT which pages to fetch - be generous, allow multiple ranges
            page_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document navigator. Given a document structure "
                            "and a topic, return up to 3 page ranges most likely to "
                            "contain relevant content. Format: '1-3,7-9,15-17' "
                            "Use comma separation for multiple ranges. "
                            "If truly nothing is relevant return NONE. "
                            "Be generous — if unsure, include the pages."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Document has {page_count} pages total.\n"
                            f"Document structure:\n{structure}\n\n"
                            f"Find pages relevant to topic: '{section_title}'\n"
                            f"Keywords: {keywords}\n"
                            "Return page ranges or NONE."
                        ),
                    },
                ],
                max_tokens=50,
            )
            page_ranges_str = (page_response.choices[0].message.content or "").strip()
            print(f"  [{doc_id[:8]}] page ranges for '{section_title}': {page_ranges_str}")

            if not page_ranges_str or page_ranges_str.upper() == "NONE":
                # Fallback: pull early, middle, and later pages to avoid thin retrieval.
                page_ranges_str = "1-3,4-6,7-9"
                print(f"  [{doc_id[:8]}] fallback page ranges: {page_ranges_str}")

            # Process each range
            ranges = [r.strip() for r in page_ranges_str.split(",")]
            for page_range in ranges[:3]:  # max 3 ranges per doc
                try:
                    # Limit each range to 3 pages
                    if "-" in page_range:
                        parts = page_range.split("-")
                        start = int(parts[0].strip())
                        end = min(int(parts[1].strip()), start + 2)
                        page_range = f"{start}-{end}"

                    print(f"Attempting to fetch doc: {doc_id[:8]} pages: {page_range}")
                    try:
                        content = client.get_page_content(doc_id, page_range)
                        print(f"Content received: {type(content)} length: {len(str(content))}")
                        print(f"Content preview: {str(content)[:200]}")
                    except Exception as e:
                        print(f"get_page_content FAILED: {e}")
                        raise
                    content_str = str(content)[:2500]

                    if content_str and len(content_str) > 100:
                        retrieved_passages.append(
                            {
                                "doc_id": doc_id,
                                "pages": page_range,
                                "text": content_str,
                                "search_type": "semantic",
                            }
                        )
                        answered_aspects.append(
                            f"Retrieved from {doc_id[:8]} pages {page_range}"
                        )

                        keyword_matches = keyword_search_pages(content_str, keywords)
                        if keyword_matches:
                            retrieved_passages.append(
                                {
                                    "doc_id": doc_id,
                                    "pages": page_range,
                                    "text": " ".join(keyword_matches),
                                    "search_type": "keyword",
                                }
                            )
                            answered_aspects.append(
                                f"Keyword matches from {doc_id[:8]} pages {page_range}"
                            )
                except Exception as e:
                    gaps.append(f"Error fetching {doc_id[:8]} pages {page_range}: {str(e)}")

        except Exception as e:
            gaps.append(f"Error processing {doc_id[:8]}: {str(e)}")
            continue

    if not retrieved_passages:
        gaps.append(f"No content found for section: {section_title}")

    return {
        "section_title": section_title,
        "retrieved_passages": retrieved_passages,
        "answered_aspects": answered_aspects,
        "gaps": gaps,
    }


def route_docs_to_section(section, all_doc_descriptions) -> list:
    # Use all available indexed docs for every section.
    # For a small demo corpus this avoids brittle routing failures.
    return [str(d["doc_id"]) for d in all_doc_descriptions if d.get("doc_id")]


def run_full_retrieval():
    load_dotenv()

    registry = _load_json(REGISTRY_PATH)
    registry = [r for r in registry if r.get("status") == "indexed"]
    indexed_doc_ids = {str(r.get("doc_id")) for r in registry if r.get("doc_id")}

    outline = _load_json(OUTLINE_PATH)
    topic = outline.get("topic", "Unknown Topic")
    agent_state = _load_json(AGENT_STATE_PATH)
    all_doc_descriptions = _load_json(DESCRIPTIONS_PATH)
    all_doc_descriptions = [
        row for row in all_doc_descriptions if str(row.get("doc_id")) in indexed_doc_ids
    ]
    results = _load_json(RESULTS_PATH, default=[])

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    client = PageIndexClient(workspace="./workspace")

    session_tree_path = STATE_DIR / "session_tree.json"
    if not session_tree_path.exists():
        tree = session_memory.init_session(topic)
    else:
        tree = session_memory.load_session()

    completed = set(agent_state.get("completed_sections", []))
    sections = _flatten_sections(outline)

    for section in tqdm(sections, desc="Retrieving sections", unit="section"):
        section_title = section.get("section_title", "Untitled Section")
        if section_title in completed:
            continue

        scoped_doc_ids = route_docs_to_section(section, all_doc_descriptions)
        try:
            result = retrieve_for_section(section, scoped_doc_ids, client, tree)
        except Exception as e:
            print(f"Section '{section['section_title']}' failed: {e}")
            result = {
                "section_title": section["section_title"],
                "retrieved_passages": [],
                "answered_aspects": [],
                "gaps": [f"Retrieval failed: {str(e)}"],
            }
        results.append(result)
        _save_json(RESULTS_PATH, results)

    print(f"Retrieval complete. Sections tracked in results: {len(results)}")


if __name__ == "__main__":
    run_full_retrieval()
