from .llm_model import LLMModel
from .rag_model import RAGModel
from .deberta_model import DebertaModel

def load_model(config):
    """
    Load a model according to config. 
    config.model_type should be one of:
        - "llm"
        - "rag"
        - "deberta"
    """

    model_type = getattr(config, "model_type", None)

    if model_type == "llm":
        return LLMModel(config)
    elif model_type == "rag":
        return RAGModel(config)
    elif model_type == "deberta":
        return DebertaModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")