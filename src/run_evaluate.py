import argparse
import json
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def score_prediction(pred: str, gold: str) -> float:
    """
    Compute score for one instance:
       Full match (exact set match) → 1.0
       Partial overlap → 0.5
       No overlap → 0.0
    """
    pred_set = set(pred.split(",")) if pred else set()
    gold_set = set(gold.split(",")) if gold else set()

    if pred_set == gold_set:
        return 1.0
    elif len(pred_set & gold_set) > 0:
        return 0.5
    else:
        return 0.0


def evaluate(gold_path: str, pred_path: str):
    """Evaluate predictions against gold labels."""
    gold_data = load_jsonl(gold_path)
    pred_data = load_jsonl(pred_path)

    # Build a mapping from topic_id to its corresponding gold answer
    gold_dict = {}
    for item in gold_data:
        topic_id = str(item["topic_id"])

        gold_answer = item.get("answer") or item.get("golden_answer", "")
        gold_dict[topic_id] = gold_answer

    # Iterate through the prediction file
    total = 0
    full, partial, wrong = 0, 0, 0
    total_score = 0.0

    for pred_item in pred_data:
        topic_id = str(pred_item["topic_id"])
        pred_answer = pred_item["prediction"]
        gold_answer = gold_dict.get(topic_id)

        if gold_answer is None:
            print(f" Warning: topic_id {topic_id} not found in gold file.")
            continue

        score = score_prediction(pred_answer, gold_answer)
        total_score += score
        total += 1

        if score == 1.0:
            full += 1
        elif score == 0.5:
            partial += 1
        else:
            wrong += 1

    # Overall result
    avg_score = total_score / total if total else 0.0

    print("\nEvaluation Results")
    print(f"Total instances: {total}")
    print(f"Full matches : {full} ({full/total:.2%})")
    print(f"Partial matches : {partial} ({partial/total:.2%})")
    print(f"Wrong answers : {wrong} ({wrong/total:.2%})")
    print(f"Average score : {avg_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AER model predictions")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold questions.jsonl")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction file (submission.jsonl)")
    args = parser.parse_args()

    evaluate(args.gold, args.pred)


if __name__ == "__main__":
    main()