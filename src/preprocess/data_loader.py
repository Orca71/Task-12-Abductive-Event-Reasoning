import json
from typing import Dict, List, Tuple


def load_questions(questions_path: str) -> List[Dict]:
    """
    Load all question instances from a JSONL file.

    Each line is a JSON object with:
      - topic_id (int)
      - question (str)
      - option_A ... option_D (str)
      - answer (str) [may be absent in test/eval]
    """
    questions = []
    with open(questions_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                instance = json.loads(line)
                questions.append(instance)
            except json.JSONDecodeError:
                continue
    return questions


def load_documents(docs_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Load all documents from a JSON file and build a mapping:
        topic_id (str) → list of documents [{title, text}, ...]

    docs.json format:
    {
      "topic_id": 101,
      "docs": [
        {"title": "...", "text": "..."},
        {"title": "...", "text": "..."}
      ]
    }
    """
    with open(docs_path, "r", encoding="utf-8") as file:
        document_items = json.load(file)

    topic_to_docs = {}
    for topic_entry in document_items:
        topic_id = str(topic_entry["topic_id"])
        topic_to_docs[topic_id] = topic_entry["docs"]
    return topic_to_docs


def build_context_map(document_map: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
    """
    Convert docs list into a concatenated string for each topic.
    Example output:
        {
          "101": "title1 text1 title2 text2 ...",
          "102": "..."
        }
    """
    topic_to_context = {}
    for topic_id, docs in document_map.items():
        combined_text = " ".join(
            [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in docs]
        )
        topic_to_context[topic_id] = combined_text
    return topic_to_context


def load_aer_data(
    questions_path: str, docs_path: str
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    High-level convenience loader:
    Returns (questions, context_map)
      - questions: list of dicts [{topic_id, question, option_A...}]
      - context_map: {topic_id: concatenated context string}
    """
    questions = load_questions(questions_path)
    document_map = load_documents(docs_path)
    context_map = build_context_map(document_map)
    return questions, context_map


if __name__ == "__main__":
    # Example usage
    questions, context_map = load_aer_data(
        "data/raw/train_data/questions.jsonl",
        "data/raw/train_data/docs.json"
    )

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(context_map)} topics with context")
    print(f"Example context:\n{list(context_map.items())[0]}")