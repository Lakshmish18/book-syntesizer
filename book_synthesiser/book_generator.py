import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "state"
OUTPUTS_DIR = ROOT_DIR / "outputs"

DEDUPED_RESULTS_PATH = STATE_DIR / "deduped_results.json"
OUTLINE_PATH = STATE_DIR / "canonical_outline.json"
FINAL_BOOK_PATH = OUTPUTS_DIR / "final_book.md"
SOURCE_MAP_PATH = STATE_DIR / "source_map.json"
REGISTRY_PATH = STATE_DIR / "doc_registry.json"


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _section_lookup(deduped_results):
    mapping = {}
    for item in deduped_results:
        title = item.get("section_title")
        if title:
            mapping[title] = item
    return mapping


def deduplicate_passages_by_content(passages: list, openai_client) -> list:
    if len(passages) <= 1:
        return passages

    texts = [p.get("text", "")[:500] for p in passages]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a deduplication assistant. Given a list of numbered passages, "
                    "identify groups of passages that contain the same core information. "
                    "Return a JSON list of indices to REMOVE (keep the best one from each group). "
                    "Return [] if nothing is redundant. Return only valid JSON array of integers."
                ),
            },
            {
                "role": "user",
                "content": "Passages:\n" + "\n".join([f"[{i}]: {t}" for i, t in enumerate(texts)]),
            },
        ],
        max_tokens=200,
    )

    try:
        to_remove = json.loads((response.choices[0].message.content or "").strip())
        if isinstance(to_remove, list):
            return [p for i, p in enumerate(passages) if i not in to_remove]
    except Exception:
        pass
    return passages


def generate_section(
    section_title,
    keywords,
    passages,
    prev_title,
    next_title,
    topic,
    openai_client,
    explained_concepts=None,
    temperature=0.4,
    strict_grounding=False,
) -> str:
    formatted = ""
    for i, p in enumerate(passages):
        doc_id = p.get("doc_id", "unknown")[:8]
        pages = p.get("pages", "?")
        text = p.get("text", "")[:1500]
        formatted += f"\n--- Source {i+1} [doc:{doc_id} pages:{pages}] ---\n{text}\n"

    system_msg = """You are writing one section of a technical book. 
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
    if strict_grounding:
        system_msg += (
            "\nIMPORTANT: Use ONLY information explicitly stated in the source passages. "
            "Do not add any external knowledge."
        )

    explained_text = ", ".join(list(explained_concepts or [])[:20])

    user_msg = f"""Write the section '{section_title}' for a technical book on {topic}.

Context: Previous section was '{prev_title}'. Next section is '{next_title}'.
CONCEPTS ALREADY EXPLAINED IN EARLIER SECTIONS (do not re-explain): {explained_text}
If these concepts are relevant, reference them briefly but do not explain them from scratch again.

Your sources (read all of them, then write — do not write passage by passage):
{formatted}

Requirements:
- 500 to 800 words
- Zero repeated information — if two sources say the same thing, write it once
- Every factual claim tagged with [src:doc_id:pages]
- Flowing prose that reads as one unified voice, not a patchwork of sources
- Do not start with the section title or a definition of the topic"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=2000,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def clean_text(text: str) -> str:
    text = re.sub(r"\[REVIEW:\s*", "", text)
    text = re.sub(r"(\[src:[^\]]+\])\s*\]", r"\1", text)
    return text


def detect_hallucination(section_text, passages, openai_client):
    source_text = " ".join([p.get("text", "")[:500] for p in passages])

    if not source_text.strip():
        return True

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You check if generated text is grounded in source material. "
                    "Answer only YES or NO. YES means the generated text contains "
                    "significant information NOT present in the sources (hallucination). "
                    "NO means the text is grounded in the sources."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Sources:\n{source_text}\n\nGenerated text:\n{section_text[:1000]}\n\n"
                    "Does the generated text contain significant information "
                    "NOT found in the sources?"
                ),
            },
        ],
        max_tokens=5,
    )
    answer = (response.choices[0].message.content or "").strip().upper()
    return answer == "YES"


def export_to_pdf(md_path: str, pdf_path: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        HRFlowable,
        KeepTogether,
    )
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import re
    from datetime import datetime

    # Load topic
    topic = "Computer Organization and Design"
    try:
        with open("state/canonical_outline.json", "r", encoding="utf-8") as f:
            outline = json.load(f)
            topic = outline.get("topic", topic)
    except Exception:
        pass

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
        title=topic,
        author="Synthesised Reference",
    )

    # Color palette — dark ink, off-white feel, no bright colors
    INK = colors.HexColor("#1a1a1a")
    CHAPTER_COLOR = colors.HexColor("#1a1a1a")
    SECTION_COLOR = colors.HexColor("#1a1a1a")
    RULE_COLOR = colors.HexColor("#888888")
    META_COLOR = colors.HexColor("#555555")

    # Styles — all left-aligned, serif-feeling through spacing
    title_style = ParagraphStyle(
        "BTitle",
        fontName="Times-Bold",
        fontSize=26,
        leading=32,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "BSubtitle",
        fontName="Times-Roman",
        fontSize=13,
        leading=18,
        textColor=META_COLOR,
        alignment=TA_LEFT,
        spaceAfter=4,
    )

    meta_style = ParagraphStyle(
        "BMeta",
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        textColor=META_COLOR,
        alignment=TA_LEFT,
        spaceAfter=2,
    )

    chapter_style = ParagraphStyle(
        "BChapter",
        fontName="Times-Bold",
        fontSize=15,
        leading=20,
        textColor=CHAPTER_COLOR,
        alignment=TA_LEFT,
        spaceBefore=28,
        spaceAfter=4,
        textTransform="uppercase",
        tracking=40,
    )

    section_style = ParagraphStyle(
        "BSection",
        fontName="Times-Bold",
        fontSize=12,
        leading=16,
        textColor=SECTION_COLOR,
        alignment=TA_LEFT,
        spaceBefore=18,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BBody",
        fontName="Times-Roman",
        fontSize=11,
        leading=17,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        firstLineIndent=18,
    )

    first_para_style = ParagraphStyle(
        "BFirst",
        fontName="Times-Roman",
        fontSize=11,
        leading=17,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        firstLineIndent=0,
    )

    unavailable_style = ParagraphStyle(
        "BUnavail",
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        textColor=META_COLOR,
        alignment=TA_LEFT,
        spaceAfter=8,
    )

    story = []

    # --- TITLE PAGE ---
    # Left-aligned, minimal, like a real book title page
    story.append(Spacer(1, 5 * cm))
    story.append(Paragraph(topic, title_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="40%", thickness=1.5, color=INK, hAlign="LEFT"))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "A synthesised technical reference compiled from multiple sources",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(f"{datetime.now().strftime('%B %Y')}", meta_style))
    story.append(PageBreak())

    # Read and clean markdown
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip all tags
    content = re.sub(r"\[src:[^\]]+\]", "", content)
    content = re.sub(r"\[REVIEW:[^\]]*?\]?\]?", "", content)
    content = re.sub(
        r"\[Section \'[^\']+\': insufficient[^\]]*\]",
        "[Source material insufficient for this section.]",
        content,
    )
    content = re.sub(
        r"\[Content not available[^\]]*\]",
        "[Source material insufficient for this section.]",
        content,
    )
    content = re.sub(r"  +", " ", content)
    content = re.sub(r" \.", ".", content)
    content = re.sub(r" ,", ",", content)

    lines = content.split("\n")
    chapter_count = 0
    para_buffer = []
    first_para_in_section = True

    def safe(text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def flush(first_in_section=False):
        nonlocal first_para_in_section
        if para_buffer:
            text = " ".join(para_buffer).strip()
            if text and len(text) > 3:
                s = first_para_style if first_in_section else body_style
                story.append(Paragraph(safe(text), s))
            para_buffer.clear()

    for line in lines:
        ls = line.strip()
        if not ls:
            flush(first_para_in_section)
            first_para_in_section = False
            continue

        if ls.startswith("# "):
            flush(first_para_in_section)
            chapter_count += 1
            title_text = ls[2:].strip().upper()
            if chapter_count > 1:
                story.append(PageBreak())
            # Chapter number + rule + title
            story.append(Spacer(1, 0.5 * cm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE_COLOR))
            story.append(Spacer(1, 0.15 * cm))
            story.append(
                Paragraph(
                    f"Chapter {chapter_count}",
                    ParagraphStyle(
                        "Cnum",
                        fontName="Times-Roman",
                        fontSize=9,
                        leading=12,
                        textColor=META_COLOR,
                        alignment=TA_LEFT,
                        spaceAfter=2,
                    ),
                )
            )
            story.append(Paragraph(safe(title_text), chapter_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE_COLOR))
            story.append(Spacer(1, 0.3 * cm))
            first_para_in_section = True

        elif ls.startswith("## "):
            flush(first_para_in_section)
            section_text = ls[3:].strip()
            story.append(Paragraph(safe(section_text), section_style))
            first_para_in_section = True

        elif ls.startswith("[Source material"):
            flush(first_para_in_section)
            story.append(Paragraph(safe(ls), unavailable_style))
            first_para_in_section = False

        else:
            para_buffer.append(ls)

    flush(first_para_in_section)

    # Page number footer — right aligned page number,
    # left aligned short title — exactly like academic papers
    def on_page(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.setFillColor(META_COLOR)

        page_num = canvas.getPageNumber()

        # Skip title page
        if page_num == 1:
            canvas.restoreState()
            return

        # Left: short title in italics
        canvas.setFont("Times-Italic", 8)
        canvas.drawString(LEFT, 1.4 * cm, topic[:55])

        # Right: page number
        canvas.setFont("Times-Roman", 9)
        canvas.drawRightString(PAGE_W - RIGHT, 1.4 * cm, str(page_num - 1))

        # Thin rule above footer
        canvas.setStrokeColor(RULE_COLOR)
        canvas.setLineWidth(0.3)
        canvas.line(LEFT, 1.65 * cm, PAGE_W - RIGHT, 1.65 * cm)

        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF exported: {pdf_path}")


def generate_full_book(openai_client):
    registry = _load_json(REGISTRY_PATH)
    registry = [r for r in registry if r.get("status") == "indexed"]
    indexed_doc_ids = {str(r.get("doc_id")) for r in registry if r.get("doc_id")}

    deduped_results = _load_json(DEDUPED_RESULTS_PATH)
    outline = _load_json(OUTLINE_PATH)

    by_section = _section_lookup(deduped_results)
    ordered_sections = []
    for chapter in outline.get("chapters", []):
        for section in chapter.get("sections", []):
            ordered_sections.append(section)

    topic = outline.get("topic", "Unknown Topic")
    generated_count = 0
    skipped_count = 0
    thin_sections = []
    flagged_sentences = []
    source_map = {}
    book_parts = []
    src_tag_count = 0
    explained_concepts = set()

    global_idx = 0
    for chapter in outline.get("chapters", []):
        chapter_title = chapter.get("chapter_title", "Untitled Chapter")
        book_parts.append(f"# {chapter_title}\n")

        for section in chapter.get("sections", []):
            title = section.get("section_title", "Untitled Section")
            keywords = section.get("keywords", [])
            prev_title = ordered_sections[global_idx - 1]["section_title"] if global_idx > 0 else "None"
            next_title = (
                ordered_sections[global_idx + 1]["section_title"]
                if global_idx + 1 < len(ordered_sections)
                else "None"
            )
            global_idx += 1

            section_payload = by_section.get(title, {})
            passages = section_payload.get("retrieved_passages", [])
            passages = [p for p in passages if str(p.get("doc_id", "")) in indexed_doc_ids]
            passages = deduplicate_passages_by_content(passages, openai_client)
            source_map[title] = sorted({str(p.get("doc_id", "unknown")) for p in passages})

            book_parts.append(f"## {title}\n")
            if not passages:
                skipped_count += 1
                note = f"[Section '{title}': insufficient source material found]"
                book_parts.append(note + "\n")
                continue

            generated = generate_section(
                section_title=title,
                keywords=keywords,
                passages=passages,
                prev_title=prev_title,
                next_title=next_title,
                topic=topic,
                openai_client=openai_client,
                explained_concepts=explained_concepts,
            )
            hallucinated = detect_hallucination(generated, passages, openai_client)
            if hallucinated and len(passages) < 2:
                gaps = section_payload.get("gaps", []) if isinstance(section_payload, dict) else []
                generated = (
                    f"[Section '{title}': insufficient verified source material — "
                    f"gaps: {', '.join(gaps[:2])}]"
                )
            elif hallucinated:
                generated = generate_section(
                    section_title=title,
                    keywords=keywords,
                    passages=passages,
                    prev_title=prev_title,
                    next_title=next_title,
                    topic=topic,
                    openai_client=openai_client,
                    explained_concepts=explained_concepts,
                    temperature=0.1,
                    strict_grounding=True,
                )

            checked = clean_text(generated)
            generated_count += 1

            word_count = len(re.findall(r"\b\w+\b", checked))
            if word_count < 200:
                thin_sections.append({"section_title": title, "word_count": word_count})

            src_tag_count += len(re.findall(r"\[src:[^\]]+\]", checked))
            flagged_sentences.extend(re.findall(r"\[REVIEW:\s*(.*?)\]", checked, flags=re.DOTALL))

            concept_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "List the 3-5 main technical concepts explained in this text. "
                            "Return as a JSON list of short strings (2-4 words each)."
                        ),
                    },
                    {"role": "user", "content": checked[:1000]},
                ],
                max_tokens=100,
            )
            try:
                new_concepts = json.loads((concept_response.choices[0].message.content or "").strip())
                if isinstance(new_concepts, list):
                    explained_concepts.update([str(c) for c in new_concepts if isinstance(c, str)])
            except Exception:
                pass

            book_parts.append(checked + "\n")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_BOOK_PATH.write_text("\n".join(book_parts).strip() + "\n", encoding="utf-8")
    SOURCE_MAP_PATH.write_text(json.dumps(source_map, indent=2, ensure_ascii=False), encoding="utf-8")
    export_to_pdf(str(FINAL_BOOK_PATH), str(OUTPUTS_DIR / "final_book.pdf"))

    print(f"Total sections generated: {generated_count}, skipped: {skipped_count}")
    print(f"Sections under 200 words (thin): {thin_sections if thin_sections else 'none'}")
    print(f"Total [src:...] tags in book: {src_tag_count}")
    if flagged_sentences:
        print("Flagged [REVIEW:...] sentences:")
        for sentence in flagged_sentences:
            print(f"- {sentence.strip()}")
    else:
        print("No [REVIEW:...] sentences flagged.")


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env before generation.")
    client = OpenAI(api_key=api_key)
    generate_full_book(client)
