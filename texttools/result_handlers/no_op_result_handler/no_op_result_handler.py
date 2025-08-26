from typing import Any

from texttools.result_handlers import BaseResultHandler


class NoOpResultHandler(BaseResultHandler):
    """
    A result handler that does nothing!
    Useful as a default when no handler is provided.
    """

    def handle(self, results: dict[str, Any]) -> None:
        pass
