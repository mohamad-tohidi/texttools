from typing import Any

from texttools.result_handlers import BaseResultHandler


class PrintResultHandler(BaseResultHandler):
    """
    A result handler that prints the results to the console.
    Useful for debugging or local tests.
    """

    def handle(self, results: dict[str, Any]) -> None:
        print(results["result"])
