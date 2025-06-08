from abc import ABC, abstractmethod
from typing import Dict
from enum import Enum
from elasticsearch import Elasticsearch, helpers


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

    def handle(self, results) -> None:
        pass


class PrintResultHandler(ResultHandler):
    """
    A simple handler that prints results to the console.
    Useful for debugging or local tests.
    """

    def handle(self, results) -> None:
        for key, value in results.items():
            print(f"Text ID: {key}, Category: {value.name}")


class SaveToElasticResultHandler(ResultHandler):
    """
    A simple handler that saves results to an elastic index.
    """

    def __init__(self, es_client: Elasticsearch, index_name: str):
        self.es_client = es_client
        self.index_name = index_name

    def handle(self, results):
        documents = [
            {"TextID": key, "Category": value.name} for key, value in results.items()
        ]

        actions = [{"_index": self.index_name, "_source": doc} for doc in documents]

        helpers.bulk(self.es_client, actions)
