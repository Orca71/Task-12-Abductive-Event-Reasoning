from .base_model import BaseModel

class LLMModel(BaseModel):
    """
    Zero-shot / Few-shot LLM baseline.
    """

    def __init__(self, config):
        self.config = config
        self.model_name = getattr(config, "model_name", "llama-3.1-8b-instruct")
        self.llm = None

    def load(self):
        """
        Load the LLM. For now, leave as placeholder.
        Later: load transformers model or OpenAI API wrapper.
        """
        print(f"[LLMModel] Loading model: {self.model_name}")
        self.llm = "dummy_llm"

    def build_prompt(self, item):
        """Construct a simple prompt for AER."""
        event = item["event"]
        docs = item["documents"]
        candidates = item["candidates"]

        prompt = (
            f"Event: {event}\n\n"
            f"Documents: {docs}\n\n"
            f"Candidate causes:\n"
        )
        for i, c in enumerate(candidates):
            prompt += f"{chr(65+i)}. {c}\n"

        prompt += "\nChoose the most direct cause (A/B/C...)."
        return prompt

    def predict(self, item):
        """Dummy prediction for now."""
        prompt = self.build_prompt(item)
        # TODO: call actual LLM
        return "A"  # placeholder