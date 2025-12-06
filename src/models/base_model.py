from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BaseModel:
    """
    Abstract base class defining the model interface.
    All concrete models (TF-IDF, embedding, DeBERTa, etc.)
    must implement the `predict()` method.
    """

    def predict(self, question_item: Dict, context_text: str) -> str:
        """
        Given one question and its context, predict the most plausible cause(s).

        Args:
            question_item (Dict): Contains:
                - "topic_id": str or int
                - "question": event description
                - "option_A"..."option_D": candidate causes
            context_text (str): Concatenated related documents

        Returns:
            str: Comma-separated option letters, e.g. "A,B" or "D"
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TfidfModel(BaseModel):
    """
    Simple baseline model using TF-IDF + cosine similarity.
    Chooses the option(s) most similar to the event+context text.
    """

    def __init__(self, similarity_margin: float = 0.1):
        """
        Args:
            similarity_margin (float): how close to the max similarity
                a candidate must be to be selected (default = 0.1 → within 10%)
        """
        self.similarity_margin = similarity_margin

    def predict(self, question_item: Dict, context_text: str) -> str:
        # Extract input text pieces
        event_description = (
        question_item.get("question")
        or question_item.get("target_event")
        or question_item.get("event")
        or ""
        )

        candidate_options = [
        question_item.get("option_A", ""),
        question_item.get("option_B", ""),
        question_item.get("option_C", ""),
        question_item.get("option_D", ""),
        ]


        # Combine event and context as the query
        query_text = event_description + " " + context_text
        all_texts = [query_text] + candidate_options

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Compute cosine similarity between query and each option
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Choose all options within the similarity margin of the max
        max_score = similarity_scores.max()
        best_indices = np.where(similarity_scores >= max_score * (1 - self.similarity_margin))[0]

        # Default fallback: if nothing above threshold, choose "D" (None of the others)
        if len(best_indices) == 0:
            best_indices = [3]

        predicted_letters = [chr(ord("A") + i) for i in best_indices]
        return ",".join(predicted_letters)