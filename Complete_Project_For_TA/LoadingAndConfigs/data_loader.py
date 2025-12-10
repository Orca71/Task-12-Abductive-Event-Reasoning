# aer/data_loader.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Iterable


# ============================================================
# DATA CLASS
# ============================================================
@dataclass
class QuestionExample:
    topic_id: int
    uuid: str
    target_event: str
    options: Dict[str, str]
    golden_answer: List[str]
    topic_text: str = ""        # Filled in by attach_docs()
    docs: List[str] = None      # List of cleaned snippet strings

    def __repr__(self):
        return (
            f"QEx(topic={self.topic_id}, "
            f"uuid='{self.uuid[:6]}...', "
            f"gold={self.golden_answer}, "
            f"docs={len(self.docs) if self.docs else 0})"
        )


# ============================================================
# LOAD DOCS.JSON (TRAIN / DEV)
# ============================================================
def load_docs(path: Path | str) -> Dict[int, Dict[str, Any]]:
    """
    Expected structure:
    [
      {
        "topic_id": 1,
        "topic": "...",
        "docs": [
            {"snippet": "..."},
            {"snippet": "..."},
            ...
        ]
      }
    ]

    Returns:
      {
        topic_id: {
            "topic_text": "...",
            "snippets": ["...", "...", ...]
        }
      }
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        docs_data = json.load(f)

    topic_to_docs = {}

    for item in docs_data:
        topic_id = item["topic_id"]
        topic_text = item.get("topic", "")

        snippets = []
        for d in item.get("docs", []):
            snip = d.get("snippet")
            if isinstance(snip, str) and snip.strip():
                snippets.append(snip.strip())

        topic_to_docs[topic_id] = {
            "topic_text": topic_text,
            "snippets": snippets,
        }

    return topic_to_docs


# ============================================================
# LOAD QUESTIONS.JSONL
# ============================================================
def load_questions(path: Path | str) -> List[QuestionExample]:
    path = Path(path)
    out = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            options = {
                "A": obj["option_A"],
                "B": obj["option_B"],
                "C": obj["option_C"],
                "D": obj["option_D"],
            }

            # golden may be "A" or "A,B" or ["A","B"]
            raw = obj["golden_answer"]
            if isinstance(raw, list):
                golden = raw
            else:
                golden = [x.strip() for x in raw.split(",")]

            golden = [g for g in golden if g in options]

            out.append(
                QuestionExample(
                    topic_id=obj["topic_id"],
                    uuid=obj["uuid"],
                    target_event=obj["target_event"],
                    options=options,
                    golden_answer=golden,
                )
            )

    return out


# ============================================================
# ATTACH DOCS TO QUESTIONS
# ============================================================
def attach_docs(
    questions: Iterable[QuestionExample],
    topic_to_docs: Dict[int, Dict[str, Any]]
) -> List[QuestionExample]:

    final = []
    for q in questions:
        if q.topic_id in topic_to_docs:
            q.topic_text = topic_to_docs[q.topic_id]["topic_text"]
            q.docs = topic_to_docs[q.topic_id]["snippets"]
        else:
            q.topic_text = ""
            q.docs = []
        final.append(q)
    return final


# ============================================================
# HIGH-LEVEL LOAD FUNCTION (train / dev)
# ============================================================
def load_dataset(split: str) -> List[QuestionExample]:
    base = Path(__file__).resolve().parent.parent / f"{split}_data"

    docs_path = base / "docs.json"
    q_path = base / "questions.jsonl"

    topic_docs = load_docs(docs_path)
    questions = load_questions(q_path)
    questions = attach_docs(questions, topic_docs)

    return questions
