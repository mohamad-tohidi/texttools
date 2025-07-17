# handlers.py

import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel


class ResultHandler(ABC):
    """
    Abstract base class for all result handlers.
    Implement the handle() method to define custom handling logic.
    """

    @abstractmethod
    def handle(self, results: Dict[str, Enum]) -> None:
        """
        Process the categorization results.

        Args:
            results (Dict[str, Enum]): A dictionary mapping text (or IDs) to categories.
        """
        pass


class NoOpResultHandler(ResultHandler):
    """
    A result handler that does nothing.
    Useful as a default when no other handler is provided.
    """

    def handle(self, results: Dict[str, Enum]) -> None:
        pass


class PrintResultHandler(ResultHandler):
    """
    A simple handler that prints results to the console.
    Useful for debugging or local tests.
    """

    def handle(self, results: Dict[str, Enum]) -> None:
        for key, value in results.items():
            print(f"Text ID: {key}, Category: {value.name}")


class SaveToFileResultHandler(ResultHandler):
    """
    A handler that saves each question + result pair to a CSV-like file,
    serializing whatever the result object is.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def handle(self, results: Dict[str, Any]) -> None:
        """
        Expects `results` to be a dict with at least:
          - "question": the original input text
          - "result":   the classification output (bool, BaseModel, dict, str, etc.)

        Appends one line per call:
            question_text,serialized_result
        """

        # Helper to turn anything into a JSON/string
        def serialize(val: Any) -> str:
            if isinstance(val, BaseModel):
                return val.model_dump_json()
            try:
                return json.dumps(val)
            except (TypeError, ValueError):
                return str(val)

        q = results.get("question", "")
        r = results.get("result", results)
        line = f"{q},{serialize(r)}\n"

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(line)


# You can add more handlers here as needed:
# - ElasticSearchResultHandler
# - DatabaseResultHandler
# - KafkaResultHandler
# - NATSResultHandler
# etc.
