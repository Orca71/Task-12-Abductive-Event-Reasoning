import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from aer.models.encoder.encoder import AEREncoder


MODEL_NAME = "BAAI/bge-small-en-v1.5"
TRAIN_PATH = "/workspace/encoder_train.jsonl"
CKPT_DIR = "/workspace/ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

BATCH = 16
LR = 1e-4
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.05


class AERDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        return ex["H_a"], ex["H_b"], ex["H_w"][0]


def collate(batch):
    H_a = [x[0] for x in batch]
    H_b = [x[1] for x in batch]
    H_w = [x[2] for x in batch]
    return H_a, H_b, H_w

def info_nce_diff(pos, neg, temperature):
    pos = F.normalize(pos, dim=-1)
    neg = F.normalize(neg, dim=-1)

    all_emb = torch.cat([pos, neg], dim=0)  # 2B x D

    logits = pos @ all_emb.t() / temperature

    B = pos.size(0)
    labels = torch.arange(B, device=pos.device)

    return F.cross_entropy(logits, labels)

def train():
    dataset = AERDataset(TRAIN_PATH)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate)

    model = AEREncoder(model_name=MODEL_NAME, proj_dim=256).to(DEVICE)

    for p in model.transformer.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.proj.parameters(), lr=LR)

    print("\n🔥 Starting DIFFERENCE-BASED InfoNCE training...\n")

    step = 0

    for epoch in range(EPOCHS):
        print(f"===== EPOCH {epoch+1}/{EPOCHS} =====")
        model.train()

        for H_a, H_b, H_w in loader:

            emb_a = model._encode_train(H_a, DEVICE)
            emb_b = model._encode_train(H_b, DEVICE)
            emb_w = model._encode_train(H_w, DEVICE)

            diff_pos = emb_b - emb_a
            diff_neg = emb_w - emb_a

            pos = model.proj(diff_pos)
            neg = model.proj(diff_neg)

            loss = info_nce_diff(pos, neg, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 50 == 0:
                print(f"[Step {step}] Loss: {loss.item():.6f}")

        ckpt = f"{CKPT_DIR}/epoch_{epoch+1}_diff.pt"
        torch.save({"proj": model.proj.state_dict()}, ckpt)
        print(f"✓ Saved checkpoint: {ckpt}")

    print("\n🎉 Difference-based InfoNCE Training Complete!\n")


if __name__ == "__main__":
    train()
