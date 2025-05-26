from abc import ABC, abstractmethod
from typing import Any, List, Optional

class BaseQuestionDetector(ABC):
    """
    Base class for all detectors that output a boolean (True/False).
    """

    def __init__(
        self,
        handlers: Optional[List[Any]] = None,
    ):
        self.handlers = handlers or []

    @abstractmethod
    def detect(self, text: str) -> bool:
        """
        Detect if the input text meets the condition.
        Should return True or False.
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
