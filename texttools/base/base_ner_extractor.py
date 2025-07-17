import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseNERExtractor(ABC):
    """
    Base class for all Named Entity Recognition (NER) systems.
    """

    def __init__(self, handlers: Optional[List[Any]] = None):
        """
        Initializes the BaseNERExtractor with optional result handlers.

        :param handlers: Optional list of handlers to process the NER results.
        """
        self.handlers = handlers or []

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts named entities from the input text.

        :param text: The text from which to extract entities.
        :return: A list of dictionaries, where each dictionary represents an entity
                 and typically includes 'text' and 'type' keys (e.g.,
                 [{"text": "John Doe", "type": "PERSON"}, ...]).
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess the input text before entity extraction.

        :param text: Raw input text.
        :return: Preprocessed text.
        """
        return text.strip()

    def _dispatch(
        self, entities: List[Dict[str, str]], original_text: Optional[str] = None
    ) -> None:
        """
        Sends the extracted entities to any registered result handlers.

        :param entities: The list of extracted entities.
        :param original_text: Optionally pass the original text.
        """
        result_data = {
            "entities": entities,
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
