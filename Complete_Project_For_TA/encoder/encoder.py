import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


class AEREncoder(nn.Module):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", proj_dim=256):
        super().__init__()

        # Load SentenceTransformer backbone (frozen later in train script)
        self.encoder = SentenceTransformer(model_name)

        # Transformer block is inside module "0"
        self.st_module = self.encoder._modules["0"]
        self.transformer = self.st_module.auto_model
        self.tokenizer = self.st_module.tokenizer

        hidden_dim = self.transformer.config.hidden_size  # 384 for bge-small

        # --------------------------------------------------
        # 🔥 Nonlinear projection head (Amplifies abductive signal)
        # --------------------------------------------------
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    # ---------------------------------------------------
    # Encode — used for both train & inference
    # ---------------------------------------------------
    def _encode_train(self, texts, device):
        """
        texts: list[str]
        Returns: embeddings of shape (batch, hidden_dim)
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        outputs = self.transformer(**tokens)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Mean pooling with attention mask
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)

        emb = summed / counts  # (batch, hidden_dim)
        return emb

    # ---------------------------------------------------
    # Forward through nonlinear projection head
    # ---------------------------------------------------
    def forward(self, texts, device=None):
        if device is None:
            device = next(self.parameters()).device

        emb = self._encode_train(texts, device)
        proj = self.proj(emb)              # nonlinear MLP
        proj = F.normalize(proj, dim=-1)   # unit-length vector
        return proj

    # ---------------------------------------------------
    # Single-text inference helper
    # ---------------------------------------------------
    @torch.no_grad()
    def encode_text(self, text: str):
        device = next(self.parameters()).device
        emb = self.forward([text], device=device)
        return emb[0].cpu()

    # ---------------------------------------------------
    # Load only the projection head weights
    # ---------------------------------------------------
    def load_projection(self, ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if "proj" in state:
            self.proj.load_state_dict(state["proj"])
            print(f"[AEREncoder] Loaded projection from {ckpt_path}")
        else:
            print("[WARNING] No 'proj' in checkpoint")
