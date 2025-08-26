import json
from typing import Any

from pydantic import BaseModel

from texttools.result_handlers import BaseResultHandler


class SaveToFileResultHandler(BaseResultHandler):
    """
    A result handler that saves each result to a CSV-like file,
    serializing whatever the result object is.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    # Helper to turn anything into a JSON/string
    def _serialize(value: Any) -> str:
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)

    def handle(self, results: dict[str, Any]) -> None:
        input_text = results["input_text"]
        value = results["result"]
        line = f"{input_text},{self.serialize(value)}\n"

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(line)
