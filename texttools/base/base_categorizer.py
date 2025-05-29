# base_categorizer.py

import logging

from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from enum import Enum
from texttools.handlers import ResultHandler, NoOpResultHandler


class BaseCategorizer(ABC):
    def __init__(
        self,
        categories: Enum,
        handlers: Optional[List[ResultHandler]] = None,
    ):
        """
        categories: An Enum class where each member represents a category.
        handlers: List of ResultHandler objects that will process results after categorization.
        """
        self.categories = categories
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
            except Exception as e:
                logging.error(
                    f"Handler {handler.__class__.__name__} failed", exc_info=True
                )
