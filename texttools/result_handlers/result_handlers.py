import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseResultHandler(ABC):
    """
    Base class for all result handlers.
    Implement the handle() method to define custom handling logic.
    """

    @abstractmethod
    def handle(self, results: dict[str, Any]) -> None:
        """
        Process the results.
        Args:
            results (dict[str, Any]): A dictionary mapping text to other objects (list, str, int, etc.).
        """
        pass


class NoOpResultHandler(BaseResultHandler):
    """
    A result handler that does nothing!
    Useful as a default when no handler is provided.
    """

    def handle(self, results: dict[str, Any]) -> None:
        pass


class PrintResultHandler(BaseResultHandler):
    """
    A result handler that prints the results to the console.
    Useful for debugging or local tests.
    """

    def handle(self, results: dict[str, Any]) -> None:
        print(results["result"])


#############
class SaveToFileResultHandler(BaseResultHandler):
    """
    A result handler that saves each result to a CSV-like file,
    serializing whatever the result object is.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def handle(self, results: dict[str, Any]) -> None:
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
