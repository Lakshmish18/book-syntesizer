import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = ROOT_DIR / "state"
RETRIEVAL_RESULTS_PATH = STATE_DIR / "retrieval_results.json"
DEDUPED_RESULTS_PATH = STATE_DIR / "deduped_results.json"
REGISTRY_PATH = STATE_DIR / "doc_registry.json"


def _normalise_for_similarity(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return vectors / norms


def deduplicate_section_passages(passages: list, openai_client) -> list:
    if len(passages) <= 1:
        return passages

    prepared = []
    for passage in passages:
        item = dict(passage)
        item["redundant"] = False
        prepared.append(item)

    texts = [str(p.get("text", "")).strip() for p in prepared]
    if not any(texts):
        return passages

    # Pass 1: one batched embeddings request keeps this stage fast and rate-efficient.
    emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = np.array([row.embedding for row in emb.data], dtype=np.float32)
    normalised = _normalise_for_similarity(vectors)
    similarity = normalised @ normalised.T

    ambiguous_pairs = []
    n = len(prepared)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity[i, j])
            if score > 0.92:
                left = prepared[i]
                right = prepared[j]
                if left["redundant"] or right["redundant"]:
                    continue
                if len(str(left.get("text", ""))) >= len(str(right.get("text", ""))):
                    right["redundant"] = True
                else:
                    left["redundant"] = True
            elif 0.80 <= score <= 0.92:
                ambiguous_pairs.append((i, j))

    merged_passages = []
    # Pass 2: bounded LLM merge for near-duplicates to avoid long hangs.
    for i, j in ambiguous_pairs[:10]:
        left = prepared[i]
        right = prepared[j]
        if left["redundant"] or right["redundant"]:
            continue

        prompt = (
            "These two passages cover similar ground. Merge them into one passage that keeps \n"
            "all unique information and removes the repeated parts. Return only the merged text, \n"
            "nothing else.\n\n"
            f"Passage A:\n{left.get('text', '')}\n\n"
            f"Passage B:\n{right.get('text', '')}"
        )
        response = openai_client.chat.completions.create(
            model="gpt-4",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "Return only merged text."},
                {"role": "user", "content": prompt},
            ],
        )
        merged_text = (response.choices[0].message.content or "").strip()
        if not merged_text:
            continue

        left["redundant"] = True
        right["redundant"] = True
        merged_passages.append(
            {
                "doc_id": f"{left.get('doc_id', 'unknown')}+{right.get('doc_id', 'unknown')}",
                "pages": f"{left.get('pages', 'unknown')}+{right.get('pages', 'unknown')}",
                "text": merged_text,
            }
        )

    kept = [p for p in prepared if not p.get("redundant", False)]
    for p in kept:
        p.pop("redundant", None)
    kept.extend(merged_passages)
    return kept


def deduplicate_all(openai_client):
    if not RETRIEVAL_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing retrieval results: {RETRIEVAL_RESULTS_PATH}")
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {REGISTRY_PATH}")

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry = [r for r in registry if r.get("status") == "indexed"]
    indexed_doc_ids = {str(r.get("doc_id")) for r in registry if r.get("doc_id")}
    retrieval_results = json.loads(RETRIEVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    deduped_results = []

    for section_item in retrieval_results:
        section_title = section_item.get("section_title", "Untitled Section")
        original_passages = section_item.get("retrieved_passages", [])
        original_passages = [p for p in original_passages if str(p.get("doc_id", "")) in indexed_doc_ids]
        if not original_passages:
            passages = []
            before = 0
            deduped_passages = []
        else:
            passages = original_passages
            before = len(passages)
            deduped_passages = deduplicate_section_passages(passages, openai_client)

            # GUARD: never reduce to zero passages
            if len(deduped_passages) == 0:
                best = max(original_passages, key=lambda p: len(p.get("text", "")))
                deduped_passages = [best]
                print(f"  Restored 1 passage for '{section_title}' (dedup would have removed all)")

        after = len(deduped_passages)
        removed = max(before - after, 0)
        print(f"{section_title}: removed {removed} passages ({before} -> {after})")

        updated = dict(section_item)
        updated["retrieved_passages"] = deduped_passages
        deduped_results.append(updated)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DEDUPED_RESULTS_PATH.write_text(
        json.dumps(deduped_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return deduped_results


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env before deduplication.")
    client = OpenAI(api_key=api_key)
    deduplicate_all(client)
