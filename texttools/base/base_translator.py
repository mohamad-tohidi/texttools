from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseTranslator(ABC):
    """
    Base class for all translators that output a translated string.
    """

    def __init__(
        self,
        handlers: Optional[List[Any]] = None,
    ):
        self.handlers = handlers or []

    @abstractmethod
    def translate(
        self, text: str, target_language: str, source_language: Optional[str] = None
    ) -> str:
        """
        Translate the input text from the source language to the target language.
        Should return the translated string.
        The source_language can be optional if the LLM can detect it automatically.
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
