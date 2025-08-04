from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseKeywordExtractor(ABC):
    """
    Base class for all detectors that output a list of keywords.
    """

    def __init__(
        self,
        handlers: Optional[list[Any]] = None,
    ):
        self.handlers = handlers or []

    @abstractmethod
    def extract_keywords(self, text: str) -> list[str]:
        """
        Extract keywords from the input text.
        Should return a list of strings, where each string is a keyword.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional text preprocessing step.
        """
        return text.strip()

    def _dispatch(self, result: dict) -> None:
        """
        Dispatch the result to handlers.
        """
        for handler in self.handlers:
            handler.handle(result)
