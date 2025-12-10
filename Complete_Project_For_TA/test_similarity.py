import torch
import torch.nn.functional as F
from aer.models.encoder.encoder import AEREncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AEREncoder().to(DEVICE)
ckpt_path = "/workspace/ckpts/epoch_5_pairwise.pt"
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.proj.load_state_dict(ckpt["proj"])
print("✓ Loaded pairwise projection from:", ckpt_path)


TEST_SET = [
    {
        "topic": "Yellow Vests",
        "H_a": "Protestors blocked the Champs Elysées because a nationwide fuel tax sparked widespread anger.",
        "H_b": "Reports indicate large-scale demonstrations across France over rising fuel prices.",
        "H_w": "Protestors blocked the Champs Elysées because the French national football team won a major match."
    },
    {
        "topic": "Myanmar Coup",
        "H_a": "Hospitals shut down because doctors went on strike after Myanmar’s military seized power.",
        "H_b": "Reports show medical workers refusing to work as part of a civil disobedience movement.",
        "H_w": "Hospitals shut down because a massive cyclone made landfall in the Ayeyarwady region."
    },
    {
        "topic": "Texas Freeze",
        "H_a": "Millions lost power in Texas because extreme cold caused failures in natural gas and wind infrastructure.",
        "H_b": "ERCOT reported widespread outages from frozen pipelines and turbines.",
        "H_w": "Millions lost power in Texas because a ransomware attack targeted the state’s grid controllers."
    }
]

PREFIX = " [CAUSE] "

for item in TEST_SET:
    topic = item["topic"]
    a, b, w = item["H_a"], item["H_b"], item["H_w"]

    P_ab = a + PREFIX + b
    P_aw = a + PREFIX + w

    with torch.no_grad():
        emb_pos = model.forward([P_ab], DEVICE)[0]
        emb_neg = model.forward([P_aw], DEVICE)[0]

    sim = F.cosine_similarity(emb_pos, emb_neg, dim=0).item()

    print(f"\n=== {topic} ===")
    print("sim(P_ab, P_aw) =", sim)
    print("Expected: LOW if abductive separation is good")
