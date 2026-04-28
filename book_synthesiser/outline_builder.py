import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pageindex.client import PageIndexClient


def _load_registry(registry_path: Path):
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing file: {registry_path}")
    return json.loads(registry_path.read_text(encoding="utf-8"))


def extract_json_from_response(text: str) -> dict:
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

    raise ValueError("No valid JSON object found in response")


def _walk_tree(node, doc_id, filename, depth=0):
    rows = []
    if isinstance(node, dict):
        title = node.get("title")
        summary = node.get("summary")
        if title or summary:
            rows.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "node_title": title or "",
                    "node_summary": summary or "",
                    "depth": depth,
                }
            )

        children = None
        for key in ["children", "nodes", "sections", "subsections"]:
            if key in node:
                children = node[key]
                break

        if isinstance(children, list):
            for child in children:
                rows.extend(_walk_tree(child, doc_id, filename, depth + 1))
        elif isinstance(children, dict):
            rows.extend(_walk_tree(children, doc_id, filename, depth + 1))

    elif isinstance(node, list):
        for item in node:
            rows.extend(_walk_tree(item, doc_id, filename, depth))

    return rows


def _build_prompt_payload(doc_descriptions, tree_rows):
    survey_payload = {
        "doc_descriptions": doc_descriptions,
        "nodes": tree_rows,
    }
    tree_survey_json = json.dumps(survey_payload, ensure_ascii=False)
    n = len(doc_descriptions)
    return (
        f"Here are section titles and summaries from {n} source documents on computer \n"
        "organization and memory systems. Generate a DETAILED canonical outline with \n"
        "at least 5 chapters and 15-20 sections total. Be specific — if the sources \n"
        "cover cache mapping techniques, that deserves its own section. If they cover \n"
        "emerging memory technologies like NVM and HBM, those deserve their own sections.\n"
        "Do not merge distinct topics into vague umbrella sections.\n\n"
        "Source data:\n"
        f"{tree_survey_json}\n\n"
        "Return the JSON outline with at minimum these chapters based on what is \n"
        "actually in the sources:\n"
        "- Basic Computer Organization (functional units, registers, bus structure)\n"
        "- Computer Architecture (ALU, microoperations, instruction types)\n"
        "- Memory Hierarchy (primary, secondary, cache levels)\n"
        "- Cache Memory (mapping, hit ratio, replacement policies, performance)\n"
        "- Emerging Memory Technologies (NVM, HBM, persistent memory, quantum memory)\n\n"
        "Each chapter should have 3-5 specific sections. Coverage count must reflect \n"
        "actual overlap across the 3 source documents.\n\n"
        "Return this exact JSON structure:\n"
        "{\n"
        "  'topic': '<inferred topic>',\n"
        "  'chapters': [\n"
        "    {\n"
        "      'chapter_title': '...',\n"
        "      'sections': [\n"
        "        {\n"
        "          'section_title': '...',\n"
        "          'keywords': ['...', '...'],\n"
        "          'coverage_count': <how many source docs cover this>,\n"
        "          'coverage_level': 'high|medium|thin',\n"
        "          'supplementary': false\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )


def _request_outline(client: OpenAI, user_content: str):
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are helping build a synthesised book. You have been given section titles \n"
                    "and summaries from multiple short source documents on the same topic. Group them into \n"
                    "a clean canonical chapter and section outline for one final book. Be practical — \n"
                    "if many documents cover the same concept, that concept belongs in the final book. \n"
                    "If only one document mentions something niche, mark it supplementary. \n"
                    "Return only valid JSON, no explanation, no markdown fences."
                ),
            },
            {"role": "user", "content": user_content},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _count_sections(outline):
    total = 0
    for chapter in outline.get("chapters", []):
        total += len(chapter.get("sections", []))
    return total


def _print_outline(outline):
    print(f"Topic: {outline.get('topic', 'Unknown Topic')}")
    for c_idx, chapter in enumerate(outline.get("chapters", []), start=1):
        print(f"Chapter {c_idx}: {chapter.get('chapter_title', 'Untitled Chapter')}")
        for s_idx, section in enumerate(chapter.get("sections", []), start=1):
            title = section.get("section_title", "Untitled Section")
            level = section.get("coverage_level", "unknown")
            count = section.get("coverage_count", 0)
            print(f"  {c_idx}.{s_idx} {title} (coverage: {level}, {count} docs)")


def build_outline():
    root_dir = Path(__file__).resolve().parent
    state_dir = root_dir / "state"
    registry_path = state_dir / "doc_registry.json"
    survey_path = state_dir / "tree_survey.json"
    outline_path = state_dir / "canonical_outline.json"
    agent_state_path = state_dir / "agent_state.json"

    state_dir.mkdir(parents=True, exist_ok=True)

    registry = _load_registry(registry_path)
    registry = [r for r in registry if r.get("status") == "indexed"]

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    client = PageIndexClient(workspace="./workspace")

    tree_rows = []
    doc_descriptions = []
    for row in registry:
        if row.get("status") != "indexed":
            continue
        doc_id = row.get("doc_id")
        filename = row.get("filename", "unknown.pdf")
        if not doc_id:
            continue

        metadata = client.get_document(doc_id)
        description = metadata.get("description") if isinstance(metadata, dict) else None
        if description:
            doc_descriptions.append(str(description))
        else:
            doc_descriptions.append(f"{filename} (no description)")

        structure = client.get_document_structure(doc_id)
        tree_rows.extend(_walk_tree(structure, doc_id=doc_id, filename=filename, depth=0))

    survey_payload = {
        "doc_descriptions": doc_descriptions,
        "nodes": tree_rows,
    }
    survey_path.write_text(json.dumps(survey_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_prompt = _build_prompt_payload(doc_descriptions, tree_rows)

    raw = _request_outline(openai_client, user_prompt)
    try:
        outline = extract_json_from_response(raw)
    except ValueError:
        raw_retry = _request_outline(openai_client, user_prompt)
        outline = extract_json_from_response(raw_retry)

    outline_path.write_text(json.dumps(outline, indent=2, ensure_ascii=False), encoding="utf-8")

    agent_state = {
        "topic": outline.get("topic", "Unknown Topic"),
        "total_sections": _count_sections(outline),
        "completed_sections": [],
        "in_progress": None,
        "gap_log": [],
        "source_attribution": {},
    }
    agent_state_path.write_text(json.dumps(agent_state, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_outline(outline)


if __name__ == "__main__":
    load_dotenv()
    build_outline()
