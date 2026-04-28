import json
import os
from pathlib import Path

from dotenv import load_dotenv

from pageindex.client import PageIndexClient


def _build_name_to_doc_id(documents):
    mapping = {}
    for doc_id, info in documents.items():
        name = info.get("name")
        if name:
            mapping[name] = doc_id
    return mapping


def index_all_pdfs():
    root_dir = Path(__file__).resolve().parent
    pdfs_dir = root_dir / "pdfs"
    state_dir = root_dir / "state"
    registry_path = state_dir / "doc_registry.json"
    descriptions_path = state_dir / "doc_descriptions.json"

    state_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Workspace auto-loads prior indexing state, enabling resumable runs by filename check.
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    client = PageIndexClient(workspace="./workspace")

    pdf_files = sorted(pdfs_dir.glob("*.pdf"))
    name_to_doc_id = _build_name_to_doc_id(client.documents)

    registry = []
    indexed_count = 0
    skipped_count = 0
    failed_count = 0
    failed_files = []

    for pdf_path in pdf_files:
        filename = pdf_path.name
        existing_doc_id = name_to_doc_id.get(filename)

        if existing_doc_id:
            print(f"already indexed: {filename}")
            skipped_count += 1
            registry.append(
                {
                    "filename": filename,
                    "doc_id": existing_doc_id,
                    "status": "indexed",
                }
            )
            continue

        try:
            doc_id = client.index(str(pdf_path))
            indexed_count += 1
            registry.append(
                {
                    "filename": filename,
                    "doc_id": doc_id,
                    "status": "indexed",
                }
            )
            name_to_doc_id[filename] = doc_id
        except Exception as error:
            failed_count += 1
            failed_files.append(filename)
            registry.append(
                {
                    "filename": filename,
                    "doc_id": None,
                    "status": "failed",
                    "error": str(error),
                }
            )
            print(f"failed: {filename} -> {error}")

    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

    descriptions = []
    for item in registry:
        if item.get("status") != "indexed":
            continue
        doc_id = item.get("doc_id")
        if not doc_id:
            continue
        try:
            metadata = client.get_document(doc_id)
            descriptions.append(
                {
                    "filename": item["filename"],
                    "doc_id": doc_id,
                    "metadata": metadata,
                }
            )
        except Exception as error:
            descriptions.append(
                {
                    "filename": item["filename"],
                    "doc_id": doc_id,
                    "metadata": None,
                    "error": str(error),
                }
            )

    descriptions_path.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Indexing complete:")
    print(f"  Successfully indexed: {indexed_count} PDFs")
    print(f"  Already indexed (skipped): {skipped_count} PDFs")
    print(f"  Failed (will be excluded from pipeline): {failed_count} PDFs")
    print(f"  Failed files: {failed_files}")


if __name__ == "__main__":
    load_dotenv()
    index_all_pdfs()
