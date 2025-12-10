import torch 
from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class HypothesisSet:
    h_a: str
    h_b: str
    h_wrong: List[str]
    emb_a: np.ndarray = None
    emb_b: np.ndarray = None
    emb_wrong: List[np.ndarray] = None

class HypothesisEncoder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        print(f"[Encoder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(
            texts,
            convert_to_numpy = True,
            normalize_embeddings= True
        )
        return emb

def encode_hypothesis_set(
    encoder: HypothesisEncoder,
    h_a: str,
    h_b: str,
    h_wrong: List[str]
) -> HypothesisSet:
    emb_a = encoder.encode(h_a)[0]
    emb_b = encoder.encode(h_b)[0]
    emb_wrong = (
        encoder.encode(h_wrong).tolist()
        if len(h_wrong) > 0 else []
    )   
    return HypothesisSet(
        h_a=h_a,
        h_b=h_b,
        h_wrong=h_wrong,
        emb_a=emb_a,
        emb_b=emb_b,
        emb_wrong=emb_wrong
    )