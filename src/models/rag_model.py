from .base_model import BaseModel

class RAGModel(BaseModel):
    """
    Retrieval-Augmented Generation baseline.
    """

    def __init__(self, config):
        self.config = config
        self.retriever = None
        self.generator = None

    def load(self):
        """Load retriever + generator."""
        print("[RAGModel] Loading RAG components...")
        self.retriever = "dummy_retriever"
        self.generator = "dummy_generator"

    def retrieve(self, item):
        """Dummy retrieval."""
        return item["documents"][:2]  # just take first 2 docs

    def predict(self, item):
        """Use RAG pipeline."""
        retrieved_docs = self.retrieve(item)
        # TODO: combine with generator
        return "A"