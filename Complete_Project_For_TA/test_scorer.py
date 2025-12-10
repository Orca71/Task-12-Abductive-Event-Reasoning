# test_scorer.py

import torch
from aer.models.scorer.scorer import HypothesisScorer

def main():
    scorer = HypothesisScorer()

    # Fake embeddings for smoke test
    h_a = torch.randn(1, 256)
    h_b = torch.randn(1, 256)

    score = scorer(h_a, h_b)
    print("Score:", score.item())

if __name__ == "__main__":
    main()
