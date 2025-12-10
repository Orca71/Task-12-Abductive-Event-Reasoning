from aer.data_loader import load_dataset
from aer.hypothesis.builder import build_training_dataset

def main():
    train = load_dataset("train")

    dataset = build_training_dataset(train)

    print("Total built training items:", len(dataset))
    print("\n=== SAMPLE ITEM ===")
    sample = dataset[0]

    print("UUID:", sample["uuid"])
    print("Topic:", sample["topic_id"])
    print("\nH_a:", sample["H_a"])
    print("\nH_b:", sample["H_b"])
    print("\nH_wrong:")
    for w in sample["H_wrong"]:
        print("  -", w)


if __name__ == "__main__":
    main()
