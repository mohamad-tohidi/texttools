import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from texttools.handlers import NoOpResultHandler, ResultHandler


class BaseCategorizer(ABC):
    def __init__(
        self,
        handlers: Optional[List[ResultHandler]] = None,
    ):
        """
        handlers: List of ResultHandler objects that will process results after categorization.
        """
        self.handlers = handlers or [NoOpResultHandler()]

    @abstractmethod
    def categorize(self, text: str) -> Enum:
        """
        Categorize the input text.
        Must return one of the Enum members defined in self.categories.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess text before categorization.
        """
        return text

    def _dispatch(self, results: dict) -> None:
        for handler in self.handlers:
            try:
                handler.handle(results)
            except Exception:
                logging.error(
                    f"Handler {handler.__class__.__name__} failed", exc_info=True
                )
