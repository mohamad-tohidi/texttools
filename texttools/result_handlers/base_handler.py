from typing import Any
from abc import ABC, abstractmethod


class BaseResultHandler(ABC):
    """
    Base class for all result handlers.
    Implement the handle() method to define custom handling logic.
    """

    @abstractmethod
    def handle(self, results: dict[str, Any]) -> None:
        """
        Process the results.
        Arguments: results: A dictionary mapping text to other objects (list, str, int, etc.).
        """
        pass
