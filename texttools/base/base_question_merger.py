import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional


class MergingMode(Enum):
    """
    Defines the two modes for question merging.
    """

    DEFAULT_MODE = "immediate merging"
    REASON_MODE = "merging with reasoning"


class BaseQuestionsMerger(ABC):
    """
    Base class for all systems that merges more that one question with preserving the contents.
    """

    def __init__(self, handlers: Optional[list[Any]] = None):
        """
        Initializes the BaseQuestionsMerger with optional result handlers.
        :param handlers: Optional list of handlers to process the merged results.
        """
        self.handlers = handlers or []

    @abstractmethod
    def merging_question(self, questions: list[str], mode: MergingMode) -> str:
        """
        merges the input questions based on the specified mode.

        :param question: The original questions' string as a list.
        :param mode: The MergingMode indicating how the questions should be merged.
        :return: The rephrased and merged question string.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess the input questions' text before merging.

        :param text: Raw input question's texts.
        :return: Preprocessed text.
        """
        return text.strip()

    def _dispatch(self, result_data: dict) -> None:
        """
        Sends the merged question and original questions to any registered result handlers.
        :param result_data: A dictionary containing the results (e.g., {"original_question": ..., "rewritten_question": ..., "mode": ...}).
        """
        for handler in self.handlers:
            try:
                handler.handle(result_data)
            except Exception:
                logging.error(
                    f"Handler {handler.__class__.__name__} failed", exc_info=True
                )
