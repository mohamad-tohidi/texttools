import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTaskPerformer(ABC):
    """
    Base class for common functionalities of LLM-based task performers.
    This includes features like text preprocessing and dispatching results
    to registered handlers.
    """

    def __init__(self, handlers: Optional[List[Any]] = None):
        """
        Initializes the BaseTaskPerformer with optional result handlers.

        :param handlers: An optional list of handlers to process the component's results.
        """
        self.handlers = handlers or []

    def _preprocess(self, text: str) -> str:
        """
        Preprocesses input text by stripping leading/trailing whitespace.
        This can be extended for more complex preprocessing if needed.

        :param text: The raw input text.
        :return: The preprocessed text.
        """
        return text.strip()

    @abstractmethod
    def perform(self, *args, **kwargs) -> Any:
        """
        Abstract method to be implemented by concrete task performers.
        This method will execute the primary task of the class (e.g., scoring, sorting).
        The signature of args and kwargs will vary based on the specific task.
        """
        pass

    def _dispatch(self, result_data: Dict[str, Any]) -> None:
        """
        Dispatches the component's results to any registered result handlers.
        Each handler receives a dictionary of result data.

        :param result_data: A dictionary containing the results specific to the component.
        """
        for handler in self.handlers:
            try:
                handler.handle(result_data)
            except Exception as e:
                logging.error(
                    f"Handler {handler.__class__.__name__} failed: {e}", exc_info=True
                )
