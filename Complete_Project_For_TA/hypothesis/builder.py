# aer/hypothesis/builder.py

from typing import Dict, List
from aer.hypothesis.ha_gold import make_h_a, make_h_w
from aer.hypothesis.hb_docs import make_h_b
from aer.data_loader import QuestionExample


def build_training_item(q: QuestionExample) -> Dict[str, str]:
    """
    Produce a unified training item:
      - H_a : abductive gold hypothesis
      - H_b : evidence-based hypothesis
      - H_w : abductive-inverse hypothesis (list of 1 negative)
    """

    # === Hₐ (gold) ===
    h_a = make_h_a(q.target_event, q.options, q.golden_answer)

    # === H_b (from retrieved docs) ===
    h_b = make_h_b(
        target_event=q.target_event,
        topic_text=q.topic_text,
        snippets=q.docs,
    )

    # === H_w (abductive inversion of H_a) ===
    h_w_single = make_h_w(q.target_event, q.options, q.golden_answer)

    # IMPORTANT: wrap in a list for training-loop compatibility
    h_w = [h_w_single]

    return {
        "uuid": q.uuid,
        "topic_id": q.topic_id,
        "target_event": q.target_event,
        "H_a": h_a,
        "H_b": h_b,
        "H_w": h_w,   # ← always a list
    }


def build_training_dataset(examples: List[QuestionExample]) -> List[Dict[str, str]]:
    return [build_training_item(q) for q in examples]
