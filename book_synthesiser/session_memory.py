import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent
CHATINDEX_DIR = ROOT_DIR / "ChatIndex"
STATE_DIR = ROOT_DIR / "state"
SESSION_PATH = STATE_DIR / "session_tree.json"

# Use the real ChatIndex paths on disk:
# - ChatIndex/ctree/ctree.py -> CTree
# - ChatIndex/retrieval/llm_tools.py -> query_ctree
if str(CHATINDEX_DIR) not in sys.path:
    sys.path.insert(0, str(CHATINDEX_DIR))
from ctree.ctree import CTree


def _save_tree(tree):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    save_result = tree.save(str(SESSION_PATH))
    # Some implementations return a path and some return None; both are acceptable.
    _ = save_result


def _load_tree(CTree):
    if hasattr(CTree, "load") and callable(getattr(CTree, "load")):
        return CTree.load(str(SESSION_PATH))
    if hasattr(CTree, "from_json") and callable(getattr(CTree, "from_json")):
        return CTree.from_json(str(SESSION_PATH))

    # Fallback: instantiate and try instance-level load API if provided.
    tree = CTree(max_children=8)
    if hasattr(tree, "load") and callable(getattr(tree, "load")):
        loaded = tree.load(str(SESSION_PATH))
        return loaded if loaded is not None else tree

    raise RuntimeError("Could not find a compatible CTree load method for session_tree.json.")


def init_session(topic: str):
    tree = CTree(max_children=8)
    tree.add(
        [
            {
                "role": "system",
                "content": (
                    f"We are synthesising a book on the topic: {topic}. "
                    "This session tracks what has been retrieved for each section, "
                    "what information was found, and what gaps remain."
                ),
            },
            {"role": "user", "content": "Starting new book synthesis session."},
            {
                "role": "assistant",
                "content": (
                    "Session initialized. I will track all section retrievals, "
                    "what was found in each, and any gaps that could not be filled "
                    "from the source documents."
                ),
            },
        ]
    )
    _save_tree(tree)
    return tree


def load_session():
    if not SESSION_PATH.exists():
        raise RuntimeError("No session found — run init_session first")
    return _load_tree(CTree)


def record_retrieval(tree, section_title: str, what_was_found: str, gaps: list):
    gaps_text = ", ".join(gaps) if gaps else "none"
    tree.add(
        [
            {"role": "user", "content": f"Retrieving section: {section_title}"},
            {
                "role": "assistant",
                "content": f"Found: {what_was_found}\nGaps: {gaps_text}",
            },
        ]
    )
    _save_tree(tree)


def check_prior_retrieval(tree, section_title: str) -> str:
    history_lines = []
    if hasattr(tree, "conversation") and isinstance(getattr(tree, "conversation"), list):
        for msg in tree.conversation:
            role = str(msg.get("role", "unknown")).strip()
            content = str(msg.get("content", "")).strip()
            history_lines.append(f"{role}: {content}")
    elif hasattr(tree, "to_dict") and callable(getattr(tree, "to_dict")):
        payload = tree.to_dict()
        history_lines.append(str(payload))
    else:
        history_lines.append(str(tree))
    history_text = "\n".join(history_lines) if history_lines else str(tree)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are tracking what has been retrieved in a book synthesis session.",
            },
            {
                "role": "user",
                "content": (
                    f"Session history:\n{history_text}\n\n"
                    f"Has section '{section_title}' already been retrieved? "
                    "If yes, summarise what was found and any gaps. "
                    "If no, just say: No prior retrieval found for this section."
                ),
            },
        ],
        max_tokens=300,
    )
    return (response.choices[0].message.content or "").strip()


if __name__ == "__main__":
    load_dotenv()
    session_tree = init_session("Test Topic")
    record_retrieval(
        session_tree,
        section_title="Example Section",
        what_was_found="Core concepts were covered from two source documents.",
        gaps=["advanced edge case examples"],
    )
    prior = check_prior_retrieval(session_tree, "Example Section")
    print(prior)
