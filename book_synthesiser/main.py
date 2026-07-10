import sys, os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "PageIndex"))
sys.path.insert(0, os.path.join(BASE, "ChatIndex"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import re
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(BASE)
STATE_DIR = ROOT_DIR / "state"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PDFS_DIR = ROOT_DIR / "pdfs"
PIPELINE_LOG = STATE_DIR / "pipeline_log.txt"


def log_stage(message: str):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with PIPELINE_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def should_skip_stage(stage: str, resume: bool) -> bool:
    if not resume:
        return False

    if stage == "index":
        registry = _load_json(STATE_DIR / "doc_registry.json", default=[])
        return isinstance(registry, list) and len(registry) > 0
    if stage == "outline":
        return (STATE_DIR / "canonical_outline.json").exists()
    if stage == "retrieve":
        return (STATE_DIR / "retrieval_results.json").exists()
    if stage == "dedup":
        return (STATE_DIR / "deduped_results.json").exists()
    if stage == "generate":
        return (OUTPUTS_DIR / "final_book.md").exists()
    return False


def run_stage(name: str, fn, resume: bool):
    if should_skip_stage(name, resume):
        msg = f"--- Running stage: {name} --- skipped (resume)"
        print(msg)
        log_stage(msg)
        return True

    start_msg = f"--- Running stage: {name} ---"
    print(start_msg)
    log_stage(f"START {name}")
    try:
        fn()
        ok_msg = f"✓ Stage complete: {name}"
        print(ok_msg)
        log_stage(f"COMPLETE {name}")
        return True
    except Exception as error:
        fail_msg = f"✗ Stage failed: {name} — {error}"
        print(fail_msg)
        log_stage(f"FAIL {name}: {error}")
        return False


def run_test():
    issues = []

    if not PDFS_DIR.exists():
        issues.append("pdfs/ folder is missing")
    else:
        pdf_count = len(list(PDFS_DIR.glob("*.pdf")))
        if pdf_count < 1:
            issues.append("pdfs/ has no PDF files")

    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY is missing")
    if not os.getenv("ANTHROPIC_API_KEY"):
        issues.append("ANTHROPIC_API_KEY is missing")

    try:
        from pageindex.client import PageIndexClient  # noqa: F401
    except Exception as error:
        issues.append(f"PageIndex import failed: {error}")

    try:
        from ctree.ctree import CTree  # noqa: F401
        from retrieval.llm_tools import query_ctree  # noqa: F401
    except Exception as error:
        issues.append(f"ChatIndex import failed: {error}")

    if issues:
        print("System check failed:")
        for item in issues:
            print(f"- {item}")
    else:
        print("System ready")


def run_final_summary():
    outline = _load_json(STATE_DIR / "canonical_outline.json", default={}) or {}
    agent_state = _load_json(STATE_DIR / "agent_state.json", default={}) or {}
    book_text = ""
    book_path = OUTPUTS_DIR / "final_book.md"
    if book_path.exists():
        book_text = book_path.read_text(encoding="utf-8")

    topic = outline.get("topic", "Unknown")
    source_pdfs = len(list(PDFS_DIR.glob("*.pdf")))
    chapters = len(outline.get("chapters", []))
    sections = sum(len(c.get("sections", [])) for c in outline.get("chapters", []))

    thin_sections = []
    section_matches = list(
        re.finditer(r"^##\s+(.+?)\n(.*?)(?=^##\s+|\Z)", book_text, flags=re.MULTILINE | re.DOTALL)
    )
    for match in section_matches:
        title = match.group(1).strip()
        body = match.group(2).strip()
        words = len(re.findall(r"\b\w+\b", body))
        if 0 < words < 200:
            thin_sections.append(title)

    gap_count = len(agent_state.get("gap_log", [])) if isinstance(agent_state.get("gap_log", []), list) else 0

    print("============================================")
    print("Book Synthesiser — Run Complete")
    print("============================================")
    print(f"Topic: {topic}")
    print(f"Source PDFs: {source_pdfs}")
    print(f"Chapters: {chapters}")
    print(f"Sections: {sections}")
    print("Output: outputs/final_book.md")
    print(f"Thin sections: {thin_sections if thin_sections else []}")
    print(f"Gaps logged: {gap_count}")
    print("============================================")


def run_validate():
    issues = []
    retrieval_path = STATE_DIR / "retrieval_results.json"
    dedup_path = STATE_DIR / "deduped_results.json"
    book_path = OUTPUTS_DIR / "final_book.md"

    if not retrieval_path.exists():
        issues.append("Missing state/retrieval_results.json")
    if not dedup_path.exists():
        issues.append("Missing state/deduped_results.json")
    if not book_path.exists():
        issues.append("Missing outputs/final_book.md")
    else:
        text = book_path.read_text(encoding="utf-8").strip()
        if len(text) < 200:
            issues.append("outputs/final_book.md exists but appears too short")

    if issues:
        print("Validation failed:")
        for item in issues:
            print(f"- {item}")
        raise RuntimeError("Validation checks failed")

    print("Validation passed")
    print("- retrieval_results.json present")
    print("- deduped_results.json present")
    print("- final_book.md present with non-trivial content")


def parse_args():
    parser = argparse.ArgumentParser(description="Book Synthesiser pipeline runner")
    parser.add_argument(
        "--mode",
        choices=["synthesise", "reconstruct"],
        default="synthesise",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help='Required when --mode reconstruct (e.g. "Build the complete Dasharatha family tree")',
    )
    parser.add_argument(
        "--stage",
        choices=["index", "outline", "retrieve", "dedup", "generate", "validate", "all"],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--test-single",
        action="store_true",
        help="For reconstruction mode, use only first indexed cache-related document.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if args.test:
        run_test()
        return

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Lazy imports prevent any non-essential module side effects at process startup.
    import indexer
    from pageindex.client import PageIndexClient
    import outline_builder
    import retrieval_agent
    import deduplicator
    import book_generator
    import reconstruction_agent
    import graph_exporter

    if args.mode == "reconstruct":
        if not args.query.strip():
            raise ValueError("When --mode reconstruct is used, --query must be provided.")

        print("--- Running reconstruction mode ---")
        log_stage("START reconstruct")

        workspace_path = "./workspace"
        if os.path.exists(workspace_path):
            shutil.rmtree(workspace_path)
            os.makedirs(workspace_path)
            print("Workspace cleared")
        indexer.index_all_pdfs()

        # 1) reuse existing indexed docs first; index only if needed.
        registry = _load_json(STATE_DIR / "doc_registry.json", default=[]) or []
        all_doc_ids = [
            str(row.get("doc_id"))
            for row in registry
            if row.get("status") == "indexed" and row.get("doc_id")
        ]
        if all_doc_ids:
            print(f"Using {len(all_doc_ids)} existing indexed documents from state/doc_registry.json")
        else:
            print("No indexed docs found in registry, running indexing now...")
            try:
                indexer.index_all_pdfs()
            except Exception as exc:
                print(f"Indexing encountered an error: {exc}")
                print("Continuing with whatever indexed docs are available in registry.")
            registry = _load_json(STATE_DIR / "doc_registry.json", default=[]) or []
            all_doc_ids = [
                str(row.get("doc_id"))
                for row in registry
                if row.get("status") == "indexed" and row.get("doc_id")
            ]

        if not all_doc_ids:
            raise RuntimeError(
                "No indexed documents available for reconstruction. "
                "Please run `python main.py --stage index` and retry."
            )

        if args.test_single:
            cache_row = None
            for row in registry:
                if row.get("status") != "indexed" or not row.get("doc_id"):
                    continue
                name = str(row.get("filename") or row.get("path") or "")
                if "cache" in name.lower():
                    cache_row = row
                    break
            if cache_row:
                all_doc_ids = [str(cache_row.get("doc_id"))]
                print(
                    "Test-single mode enabled. Using one cache document: "
                    f"{str(cache_row.get('doc_id'))[:8]} ({cache_row.get('filename') or cache_row.get('path')})"
                )
            else:
                all_doc_ids = all_doc_ids[:1]
                print("Test-single mode enabled. No cache-named doc found, using first indexed doc.")

        # 2) page index client
        pageindex_client = PageIndexClient(workspace="./workspace")

        # 3) openai client already created above

        # 4) run reconstruction
        result = reconstruction_agent.run_reconstruction(
            query=args.query,
            doc_ids=all_doc_ids,
            client=pageindex_client,
            openai_client=openai_client,
        )

        # 5) export reconstruction graph report
        graph_exporter.export_graph_to_pdf(
            graph=result["graph"],
            query=args.query,
            validation=result["validation"],
            pdf_path="outputs/reconstruction.pdf",
            openai_client=openai_client,
        )

        # 6) completion summary
        graph = result.get("graph", {}) or {}
        validation = result.get("validation", {}) or {}
        node_count = len((graph.get("nodes") or {}).keys())
        edge_count = len(graph.get("edges") or [])
        conflict_count = 0
        for edge in graph.get("edges", []) or []:
            if edge.get("conflicting_sources"):
                conflict_count += 1

        completeness = float(validation.get("completeness_score", 0.0))
        print("============================================")
        print("Reconstruction — Run Complete")
        print("============================================")
        print(f"Query: {args.query}")
        print(f"Entities found: {node_count}")
        print(f"Relations found: {edge_count}")
        print(f"Conflicts: {conflict_count}")
        print(f"Completeness: {completeness * 100:.0f}%")
        print(f"Missing: {validation.get('missing_aspects', [])}")
        print("Output: outputs/reconstruction.pdf")
        print("============================================")
        log_stage("COMPLETE reconstruct")
        return

    stage_map = {
        "index": lambda: indexer.index_all_pdfs(),
        "outline": lambda: outline_builder.build_outline(),
        "retrieve": lambda: retrieval_agent.run_full_retrieval(),
        "dedup": lambda: deduplicator.deduplicate_all(openai_client),
        "generate": lambda: book_generator.generate_full_book(openai_client),
        "validate": lambda: run_validate(),
    }

    stage_order = ["index", "outline", "retrieve", "dedup", "generate", "validate"]
    selected = stage_order if args.stage == "all" else [args.stage]

    for stage_name in selected:
        ok = run_stage(stage_name, stage_map[stage_name], args.resume)
        if not ok:
            return

    run_final_summary()


if __name__ == "__main__":
    import re

    main()
