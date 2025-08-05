import logging
from abc import ABC, abstractmethod
from typing import Optional

from texttools.handlers import NoOpResultHandler, ResultHandler


class BaseSummarizer(ABC):
    def __init__(self, handlers: Optional[list[ResultHandler]] = None):
        """
        Base class for text summarization.

        :param handlers: Optional list of handlers to process the summarization result.
        """
        self.handlers = handlers or [NoOpResultHandler()]

    @abstractmethod
    def summarize(self, text: str) -> str:
        """
        Generate a summary for the input text.

        :param text: The text to summarize.
        :return: A summary string.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess the input text before summarization.

        :param text: Raw input text.
        :return: Preprocessed text.
        """
        return text

    def _dispatch(self, summary: str, original_text: Optional[str] = None) -> None:
        """
        Send the summary result to any registered result handlers.

        :param summary: The generated summary.
        :param original_text: Optionally pass the original text.
        """
        result_data = {
            "summary": summary,
        }
        if original_text is not None:
            result_data["original_text"] = original_text

        for handler in self.handlers:
            try:
                handler.handle(result_data)
            except Exception:
                logging.error(
                    f"Handler {handler.__class__.__name__} failed", exc_info=True
                )
