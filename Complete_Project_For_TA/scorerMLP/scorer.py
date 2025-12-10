import torch
import torch.nn as nn
import torch.nn.functional as F

class HypothesisScorer(nn.Module):
    def __init__(self, emb_dim=1024, hidden_dim=512, dropout=0.1):
        super().__init__()

        # Input is concatenation → 2 * emb_dim
        in_dim = emb_dim * 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)   # outputs scalar
        )

    def forward(self, h_a, h_b):
        """
        h_a : (batch, emb_dim)
        h_b : (batch, emb_dim)
        Returns scalar score (batch, 1)
        """
        x = torch.cat([h_a, h_b], dim=-1)
        score = self.net(x)
        return score.squeeze(-1)
