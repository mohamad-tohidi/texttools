import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional


class RewriteMode(Enum):
    """
    Defines the two modes for question rewriting.
    """

    SAME_MEANING_DIFFERENT_WORDING = "same_meaning_different_wording"
    DIFFERENT_MEANING_SIMILAR_WORDING = "different_meaning_similar_wording"


class BaseQuestionRewriter(ABC):
    """
    Base class for all systems that rewrite a question with different wording.
    """

    def __init__(self, handlers: Optional[List[Any]] = None):
        """
        Initializes the BaseQuestionRewriter with optional result handlers.

        :param handlers: Optional list of handlers to process the rewriting results.
        """
        self.handlers = handlers or []

    @abstractmethod
    def rewrite_question(self, question: str, mode: RewriteMode) -> str:
        """
        Rewrites the input question based on the specified mode.

        :param question: The original question string.
        :param mode: The RewriteMode indicating how the question should be rewritten.
        :return: The rephrased question string.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess the input question text before rewriting.

        :param text: Raw input question text.
        :return: Preprocessed text.
        """
        return text.strip()

    def _dispatch(self, result_data: dict) -> None:
        """
        Sends the rewritten question and original question to any registered result handlers.

        :param result_data: A dictionary containing the results (e.g., {"original_question": ..., "rewritten_question": ..., "mode": ...}).
        """
        for handler in self.handlers:
            try:
                handler.handle(result_data)
            except Exception:
                logging.error(
                    f"Handler {handler.__class__.__name__} failed", exc_info=True
                )
