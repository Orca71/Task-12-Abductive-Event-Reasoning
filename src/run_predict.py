import argparse
from src.preprocess.data_loader import load_aer_data
from src.models.model_loader import load_model
from src.models.model_inference import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction for Abductive Event Reasoning (AER) task"
    )

    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to questions.jsonl file (e.g., data/raw/dev_data/questions.jsonl)",
    )
    parser.add_argument(
        "--docs",
        type=str,
        required=True,
        help="Path to docs.json file (e.g., data/raw/dev_data/docs.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/submission.jsonl",
        help="Path to save predictions (default: results/submission.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tfidf",
        choices=["tfidf", "embedding", "deberta", "llm"],
        help="Model type to use for prediction (default: tfidf)",
    )
    args = parser.parse_args()

    print("Loading data...")
    questions, context_map = load_aer_data(args.questions, args.docs)

    print(f"Loaded {len(questions)} questions and {len(context_map)} context entries")

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print("Running inference...")
    run_inference(
        model=model,
        questions=questions,
        context_map=context_map,
        output_path=args.output,
    )

    print("Done! Predictions saved successfully.")


if __name__ == "__main__":
    main()