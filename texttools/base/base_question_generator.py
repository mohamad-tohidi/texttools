import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseQuestionGenerator(ABC):
    """
    Base class for all systems that generate a question from a given answer.
    """

    def __init__(self, handlers: Optional[List[Any]] = None):
        """
        Initializes the BaseQuestionGenerator with optional result handlers.

        :param handlers: Optional list of handlers to process the generation results.
        """
        self.handlers = handlers or []

    @abstractmethod
    def generate_question(self, answer: str) -> str:
        """
        Generates an appropriate question for the provided answer.

        :param answer: The answer string for which a question needs to be generated.
        :return: The generated question string.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess the input answer text before question generation.

        :param text: Raw input answer text.
        :return: Preprocessed text.
        """
        return text.strip()

    def _dispatch(self, result_data: dict) -> None:
        """
        Sends the generated question and original answer to any registered result handlers.

        :param result_data: A dictionary containing the results (e.g., {"original_answer": ..., "generated_question": ...}).
        """
        for handler in self.handlers:
            try:
                handler.handle(result_data)
            except Exception:
                logging.error(
                    f"Handler {handler.__class__.__name__} failed", exc_info=True
                )


class BaseQuestionGeneratorFromSubject(ABC):
    """
    Base class for all systems that generate a question from a given subject
    it will curate some number of questions

    """

    def __init__(self, handlers: Optional[List[Any]] = None):
        """
        Initializes the BaseQuestionGeneratorFromSubject with optional result handlers.

        :param handlers: Optional list of handlers to process the generation results.
        """
        self.handlers = handlers or []

    @abstractmethod
    def generate_question(self, subject: str) -> str:
        """
        Generates an appropriate question for the provided answer.

        :param answer: The answer string for which a question needs to be generated.
        :return: The generated question string.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess the input answer text before question generation.

        :param text: Raw input answer text.
        :return: Preprocessed text.
        """
        return text.strip()

    def _dispatch(self, result_data: dict) -> None:
        """
        Sends the generated question and original answer to any registered result handlers.

        :param result_data: A dictionary containing the results (e.g., {"original_answer": ..., "generated_question": ...}).
        """
        for handler in self.handlers:
            try:
                handler.handle(result_data)
            except Exception:
                logging.error(
                    f"Handler {handler.__class__.__name__} failed", exc_info=True
                )
