from abc import ABC, abstractmethod
from typing import Optional

from texttools.handlers import NoOpResultHandler, ResultHandler


class BaseRouter(ABC):
    def __init__(self, handlers: Optional[list[ResultHandler]] = None):
        """
        Base class for routers

        :param handlers: Optional list of handlers to process the summarization result.
        """
        self.handlers = handlers or [NoOpResultHandler()]

    @abstractmethod
    def route(self, text: str) -> str:
        """
        decides and classifies the inputted text between the choices that it has

        :param text: The text to summarize.
        :return: A route for the given text.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess the input text before summarization.

        :param text: Raw input text.
        :return: Preprocessed text.
        """
        return text.strip()
