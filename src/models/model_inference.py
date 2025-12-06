import json
import os
from typing import List, Dict
from tqdm import tqdm


def run_inference(
    model,
    questions: List[Dict],
    context_map: Dict[str, str],
    output_path: str,
) -> None:
    """
    Unified inference pipeline for all models.

    Args:
        model: an initialized model object with a .predict(question_dict, context_str) method
        questions: list of question dicts (from questions.jsonl)
        context_map: mapping topic_id → concatenated document text
        output_path: path to save the predictions in submission format
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions = []

    for question_item in tqdm(questions, desc="Running inference"):
        topic_id = str(question_item["topic_id"])
        context_text = context_map.get(topic_id, "")
        prediction_str = model.predict(question_item, context_text)

        predictions.append(
            {"topic_id": topic_id, "prediction": prediction_str}
        )

    # Save predictions
    with open(output_path, "w", encoding="utf-8") as output_file:
        for item in predictions:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Inference completed. Results saved to: {output_path}")