class BaseModel:
    """
    Base interface for all models (LLM, RAG, fine-tuned).
    """

    def load(self):
        """Load model weights/resources."""
        raise NotImplementedError

    def predict(self, item):
        """
        Run inference on one instance.
        `item` should be a dict containing event, documents, candidates, etc.
        """
        raise NotImplementedError