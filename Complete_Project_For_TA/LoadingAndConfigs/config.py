from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class DataConfig:
    train_dir: Path = PROJECT_ROOT/ "train_data"
    dev_dir: Path = PROJECT_ROOT / "dev_data"


@dataclass
class ModelConfig:
    def __init__(self):
        self.model_name = "BAAI/bge-m3"   # you already had this
        self.hidden_dim = 1024            # BGE-M3 embedding size
        self.proj_dim = 256               # your chosen projection size
        self.ckpt_path = "aer/models/encoder_ckpts/epoch_3.pt"

@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-2
    device: str = "cuda"

data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainingConfig()