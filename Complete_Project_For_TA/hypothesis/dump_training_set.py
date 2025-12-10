# aer/hypothesis/dump_training_set.py

import json
from pathlib import Path
from typing import List, Dict, Any

from aer.data_loader import load_dataset
from aer.hypothesis.builder import build_training_dataset


def dump_jsonl(data: List[Dict[str, Any]], out_path: Path | str):
    """
    Write list of dicts into a JSONL file.
    """
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(data)} items → {out_path}")


def main():
    print("Loading train split...")
    train = load_dataset("train")

    print("Building training items (H_a, H_b, H_w)...")
    items = build_training_dataset(train)

    print("Dumping encoder_train.jsonl...")
    dump_jsonl(items, "encoder_train.jsonl")

    print("\n=== DONE ===\n")
    print("Sample item keys:", list(items[0].keys()))
    print("Example item:")
    print(json.dumps(items[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
