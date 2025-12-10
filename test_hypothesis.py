# test_hypothesis.py

import torch
import torch.nn.functional as F

from aer.data_loader import load_questions, load_docs, attach_docs
from aer.hypothesis.ha_gold import make_h_a, make_h_w
from aer.hypothesis.hb_docs import make_h_b
from aer.models.inference import HypothesisEncoder


def main():
    print("=== Loading data ===")

    questions = load_questions("sample_data/questions.jsonl")
    docs = load_docs("sample_data/docs.json")
    questions = attach_docs(questions, docs)

    q = questions[0]

    print("\n=== QUESTION ===")
    print("Topic:", q.topic_id)
    print("Target Event:", q.target_event)
    print("Golden:", q.golden_answer)
    print("Options:", q.options)
    print("Num Snippets:", len(q.docs))

    # ---------------------------------------------------
    # Build hypotheses with unified templates
    # ---------------------------------------------------
    h_a = make_h_a(q.target_event, q.options, q.golden_answer)
    h_b = make_h_b(q.target_event, q.topic_text, q.docs)
    h_w = make_h_w(q.target_event, q.options, q.golden_answer)

    print("\n=== HYPOTHESES ===")
    print("\nH_a:\n", h_a)
    print("\nH_b:\n", h_b)
    print("\nH_w:\n", h_w)

    # ---------------------------------------------------
    # Load the trained encoder
    # ---------------------------------------------------
    encoder = HypothesisEncoder(ckpt_path="/workspace/ckpts/epoch_3.pt")

    # Encode embeddings
    emb_a = encoder.encode([h_a])
    emb_b = encoder.encode([h_b])
    emb_w = encoder.encode([h_w])

    # ---------------------------------------------------
    # Abductive geometry scores
    # ---------------------------------------------------
    sim_ab = F.cosine_similarity(emb_a, emb_b).item()
    sim_aw = F.cosine_similarity(emb_a, emb_w).item()

    print("\n=== ABDUCTIVE GEOMETRY ===")
    print(f"cos(H_a, H_b) = {sim_ab:.4f}   (should be HIGH)")
    print(f"cos(H_a, H_w) = {sim_aw:.4f}   (should be LOW)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
