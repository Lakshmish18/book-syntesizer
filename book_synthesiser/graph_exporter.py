import json
import re
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer


ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
STATE_DIR = ROOT_DIR / "state"


BOOK_GENERATOR_SYSTEM_MESSAGE = """You are writing one section of a technical book. 
Your writing style is clear, direct, and authoritative — like a senior engineer 
explaining something they know deeply. Not a textbook. Not a blog post.

STRICT RULES you must follow:
- Read ALL source passages first before writing a single word
- Identify what is UNIQUE in each passage vs what is repeated across sources
- Write each concept or fact EXACTLY ONCE — never repeat it
- If 3 sources say the same thing, write it once using the best explanation
- Combine complementary details from different sources into single coherent sentences
- Do NOT write an intro sentence that restates the section title
- Do NOT end with a summary paragraph that repeats what was just said
- Vary sentence length — mix short punchy sentences with longer explanatory ones
- Never use: Furthermore, Additionally, Moreover, In conclusion, In summary
- Never use bullet points
- Every factual sentence must end with a source tag: [src:doc_id:pages]
- BANNED: Do not write any paragraph that starts with 'In summary',
'In conclusion', 'To summarize', 'Overall', or 'In essence'.
End the section mid-thought if needed — do not wrap up with a summary.
The book has a conclusion chapter for that.
- If the source passages mention specific technical details like hit ratio
formulas, mapping techniques, specific memory technologies by name (NVM, HBM,
DRAM, SRAM, ReRAM), clock cycles, or performance equations — include them.
Technical specificity makes this book valuable. Do not genericise specific
technical content into vague descriptions."""


def _safe(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _group_edges_by_from(graph: dict):
    nodes = graph.get("nodes", {}) or {}
    grouped = {}
    for edge in graph.get("edges", []) or []:
        from_name = nodes.get(edge.get("from_node", ""), {}).get("canonical_name", edge.get("from_node", "unknown"))
        to_name = nodes.get(edge.get("to_node", ""), {}).get("canonical_name", edge.get("to_node", "unknown"))
        grouped.setdefault(from_name, []).append(
            {
                "relation": edge.get("relation", "related_to"),
                "to_name": to_name,
                "confidence": edge.get("confidence", 0.0),
                "sources": edge.get("sources", []),
                "conflicting_sources": edge.get("conflicting_sources", []),
            }
        )
    return grouped


def _confidence_band(value) -> str:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    if v >= 0.75:
        return "high"
    if v >= 0.45:
        return "medium"
    return "low"


def _collect_conflict_rows(graph: dict):
    nodes = graph.get("nodes", {}) or {}
    unique_relations = {"is", "equals", "defined_as", "type_is", "version_is", "also_known_as"}
    buckets = {}
    for edge in graph.get("edges", []) or []:
        relation = str(edge.get("relation", "related_to"))
        if relation.lower() not in unique_relations:
            continue
        key = (edge.get("from_node"), relation)
        buckets.setdefault(key, []).append(edge)

    rows = []
    for (from_node_id, relation), edges in buckets.items():
        to_ids = {e.get("to_node") for e in edges}
        if len(to_ids) < 2:
            continue
        entity = nodes.get(from_node_id, {}).get("canonical_name", from_node_id)
        version_edges = edges[:2]
        a, b = version_edges[0], version_edges[1]
        a_value = nodes.get(a.get("to_node", ""), {}).get("canonical_name", a.get("to_node", "unknown"))
        b_value = nodes.get(b.get("to_node", ""), {}).get("canonical_name", b.get("to_node", "unknown"))
        a_src = ", ".join(a.get("sources", []) or ["unknown"])
        b_src = ", ".join(b.get("sources", []) or ["unknown"])
        rows.append((entity, relation, a_value, a_src, b_value, b_src))
    return rows


def clean_passage_text(raw_text: str) -> str:
    raw_text = str(raw_text)
    try:
        pages = json.loads(raw_text)
        if isinstance(pages, list):
            joined = " ".join(
                str(p.get("content", "")) for p in pages if isinstance(p, dict)
            ).strip()
            if joined:
                return joined
    except Exception:
        pass
    text = re.sub(r'\{"page":\s*\d+,\s*"content":\s*"', "", raw_text)
    text = re.sub(r'"\}', "", text)
    return text.strip()


def _load_cleaned_passage_snippets(limit: int = 8) -> list:
    retrieval_path = STATE_DIR / "retrieval_results.json"
    if not retrieval_path.exists():
        return []
    try:
        rows = json.loads(retrieval_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    snippets = []
    for section in rows if isinstance(rows, list) else []:
        for passage in section.get("retrieved_passages", []) or []:
            cleaned = clean_passage_text(passage.get("text", ""))
            if len(cleaned) < 50:
                continue
            snippets.append(cleaned[:240])
            if len(snippets) >= limit:
                return snippets
    return snippets


def _load_indexed_source_rows() -> list:
    registry_path = STATE_DIR / "doc_registry.json"
    if not registry_path.exists():
        return []
    try:
        rows = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    sources = []
    for row in rows if isinstance(rows, list) else []:
        if row.get("status") != "indexed":
            continue
        doc_id = str(row.get("doc_id", "unknown"))[:8]
        filename = row.get("filename") or row.get("path") or "unknown_file"
        sources.append(f"{doc_id} - {filename}")
    return sources


def _build_styles():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BTitle",
        fontName="Times-Bold",
        fontSize=26,
        leading=32,
        textColor=colors.HexColor("#1a1a1a"),
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "BSubtitle",
        fontName="Times-Roman",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#555555"),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    chapter_style = ParagraphStyle(
        "BChapter",
        fontName="Times-Bold",
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#1a1a1a"),
        alignment=TA_LEFT,
        spaceBefore=28,
        spaceAfter=4,
        textTransform="uppercase",
        tracking=40,
    )
    body_style = ParagraphStyle(
        "BBody",
        fontName="Times-Roman",
        fontSize=11,
        leading=17,
        textColor=colors.HexColor("#1a1a1a"),
        alignment=TA_LEFT,
        spaceAfter=8,
        firstLineIndent=0,
    )
    mono_style = ParagraphStyle(
        "BMono",
        fontName="Courier",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1a1a1a"),
        alignment=TA_LEFT,
        spaceAfter=3,
    )
    return title_style, subtitle_style, chapter_style, body_style, mono_style


def export_graph_to_pdf(graph, query, validation, pdf_path, openai_client):
    # Step 1 — Generate concise prose summary from graph.
    nodes = graph.get("nodes", {}) or {}
    node_lines = []
    for node_id, node in nodes.items():
        node_lines.append(
            f"- {node.get('canonical_name', node_id)} "
            f"(type={node.get('type', 'concept')}, attributes={node.get('attributes', {})}, "
            f"sources={node.get('sources', [])})"
        )
    nodes_text = "\n".join(node_lines) if node_lines else "- none"
    if nodes_text == "- none":
        snippets = _load_cleaned_passage_snippets(limit=8)
        if snippets:
            nodes_text = "\n".join(f"- {s}" for s in snippets)

    grouped_edges = _group_edges_by_from(graph)
    edge_lines = []
    for from_name, rows in grouped_edges.items():
        edge_lines.append(f"{from_name}:")
        for row in rows:
            edge_lines.append(
                f"  - {row['relation']} -> {row['to_name']} "
                f"(confidence={row['confidence']}, sources={row['sources']})"
            )
    edges_text = "\n".join(edge_lines) if edge_lines else "- none"

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": BOOK_GENERATOR_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": (
                    f"Write a clear explanation of the following knowledge graph "
                    f"answering the query: '{query}'\n\n"
                    f"Entities found: {nodes_text}\n"
                    f"Relationships found: {edges_text}\n"
                    f"Conflicts noted: {validation.get('conflicts', [])}\n"
                    f"Missing aspects: {validation.get('missing_aspects', [])}\n\n"
                    "Write as flowing prose summary only. 250-450 words."
                ),
            },
        ],
    )
    prose = (response.choices[0].message.content or "").strip()

    # Step 2 — Structured rows for sections.
    noise = [
        "b.tech",
        "semester",
        "professor",
        "academic",
        "department",
        "textbook",
        "lecture",
        "syllabus",
        "course",
        "outcomes",
        "prerequisites",
    ]
    clean_edges = []
    for e in graph.get("edges", []) or []:
        from_name = nodes.get(e.get("from_node", ""), {}).get("canonical_name", e.get("from_node", "unknown"))
        to_name = nodes.get(e.get("to_node", ""), {}).get("canonical_name", e.get("to_node", "unknown"))
        from_l = str(from_name).lower()
        to_l = str(to_name).lower()
        if any(n in from_l or n in to_l for n in noise):
            continue
        clean_edges.append(e)

    found_rows = []
    confidence_rows = []
    source_rows = set()
    for edge in clean_edges:
        from_name = nodes.get(edge.get("from_node", ""), {}).get("canonical_name", edge.get("from_node", "unknown"))
        to_name = nodes.get(edge.get("to_node", ""), {}).get("canonical_name", edge.get("to_node", "unknown"))
        relation = edge.get("relation", "related_to")
        conf = edge.get("confidence", 0.0)
        src = ", ".join(edge.get("sources", []) or ["unknown"])
        found_rows.append(f"{from_name} -> {relation} -> {to_name} (source: {src})")
        confidence_rows.append(
            f"{from_name} -> {relation} -> {to_name} | confidence={float(conf):.2f} ({_confidence_band(conf)})"
        )
        for s in edge.get("sources", []) or []:
            source_rows.add(s)
        for s in edge.get("conflicting_sources", []) or []:
            source_rows.add(s)

    conflict_rows = _collect_conflict_rows(graph)

    # Step 3 — Build PDF using same styling approach as book_generator.export_to_pdf.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if not pdf_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = str(OUTPUTS_DIR / f"reconstruction_{ts}.pdf")

    PAGE_W, PAGE_H = A4
    LEFT = 3.8 * cm
    RIGHT = 2.5 * cm
    TOP = 2.5 * cm
    BOT = 2.5 * cm

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOT,
        title=query,
        author="Synthesised Reference",
    )

    title_style, subtitle_style, chapter_style, body_style, mono_style = _build_styles()
    story = []

    # Title page
    story.append(Spacer(1, 5 * cm))
    story.append(Paragraph(_safe(query), title_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="40%", thickness=1.5, color=colors.HexColor("#1a1a1a"), hAlign="LEFT"))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Knowledge Graph Reconstruction Report", subtitle_style))
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(_safe(datetime.now().strftime("%B %Y")), subtitle_style))
    story.append(PageBreak())

    # Section 1: What was found (organised by entity)
    story.append(Paragraph("CHAPTER 1", chapter_style))
    story.append(Paragraph("WHAT WAS FOUND", chapter_style))
    for para in [p.strip() for p in prose.split("\n\n") if p.strip()]:
        story.append(Paragraph(_safe(para), body_style))
    story.append(Spacer(1, 0.3 * cm))
    grouped = _group_edges_by_from(graph)
    if grouped:
        for entity, rows in grouped.items():
            story.append(Paragraph(_safe(entity), chapter_style))
            for row in rows:
                src = ", ".join(row.get("sources", []) or ["unknown"])
                story.append(
                    Paragraph(
                        _safe(f"- {row['relation']} -> {row['to_name']} (source: {src})"),
                        body_style,
                    )
                )
    else:
        story.append(Paragraph("No facts extracted.", body_style))

    # Section 2: Confidence levels
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 2", chapter_style))
    story.append(Paragraph("CONFIDENCE LEVELS", chapter_style))
    if confidence_rows:
        for row in confidence_rows:
            story.append(Paragraph(_safe(row), mono_style))
    else:
        story.append(Paragraph("No confidence-scored facts available.", body_style))

    # Section 3: Conflicts with both versions and sources
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 3", chapter_style))
    story.append(Paragraph("CONFLICTS", chapter_style))
    if conflict_rows:
        for entity, relation, a_value, a_src, b_value, b_src in conflict_rows:
            story.append(Paragraph(_safe(f"CONFLICT: {entity} {relation}"), body_style))
            story.append(Paragraph(_safe(f"Version A: {a_value} (source: {a_src})"), body_style))
            story.append(Paragraph(_safe(f"Version B: {b_value} (source: {b_src})"), body_style))
            story.append(Paragraph("Status: Unresolved", body_style))
            story.append(Spacer(1, 0.2 * cm))
    else:
        story.append(Paragraph("No conflicts detected.", body_style))

    # Section 4: Missing validation gaps
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 4", chapter_style))
    story.append(Paragraph("WHAT IS MISSING", chapter_style))
    missing = validation.get("missing_aspects", []) or []
    if missing:
        for m in missing:
            story.append(Paragraph(_safe(f"- {m}"), body_style))
    else:
        story.append(Paragraph("No gaps reported by validation.", body_style))

    # Section 5: Sources used
    story.append(PageBreak())
    story.append(Paragraph("CHAPTER 5", chapter_style))
    story.append(Paragraph("SOURCES USED", chapter_style))
    indexed_sources = _load_indexed_source_rows()
    if indexed_sources:
        for src in indexed_sources:
            story.append(Paragraph(_safe(f"- {src}"), body_style))
    else:
        # Fallback to graph sources if registry is unavailable.
        for node in nodes.values():
            for src in node.get("sources", []) or []:
                source_rows.add(src)
        if source_rows:
            for src in sorted(source_rows):
                story.append(Paragraph(_safe(f"- {src}"), body_style))
        else:
            story.append(Paragraph("No explicit sources captured.", body_style))

    doc.build(story)
    print(f"PDF exported: {pdf_path}")
