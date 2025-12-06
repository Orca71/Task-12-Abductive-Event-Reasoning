from typing import Any

from src.models.base_model import TfidfModel
# from src.models.rag_model import EmbeddingRetrievalModel
# from src.models.deberta_model import DebertaCausalModel
# from src.models.llm_model import LLMReasoningModel


def load_model(model_name: str, **kwargs) -> Any:
    """
    Factory function for loading different baseline or trained models.

    Args:
        model_name (str): Name of the model type to load.
        kwargs: Optional extra parameters (e.g., model path, device, etc.)

    Returns:
        An initialized model object with a unified .predict(...) interface.
    """
    model_name = model_name.lower()

    if model_name == "tfidf":
        print("Loading TF-IDF baseline model...")
        return TfidfModel(**kwargs)

    # elif model_name == "embedding":
    #     print("Loading embedding-based retrieval model...")
    #     return EmbeddingRetrievalModel(**kwargs)

    # elif model_name == "deberta":
    #     print("Loading fine-tuned DeBERTa model...")
    #     return DebertaCausalModel(**kwargs)

    # elif model_name == "llm":
    #     print("Loading large language model reasoning baseline...")
    #     return LLMReasoningModel(**kwargs)

    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # Example usage
    model = load_model("tfidf")
    print(f"Model loaded: {type(model).__name__}")