from abc import ABC, abstractmethod
from typing import Any


class BaseFormatter(ABC):
    """
    Adapter to convert a conversation into a specific LLM API's input format.

    Concrete implementations transform standardized messages (e.g., list[dict]) into the
    exact payload required by a provider (e.g., OpenAI's message list, a single string, etc.).
    """

    @abstractmethod
    def format(
        self,
        messages: Any,
    ) -> Any:
        """
        Transform the input messages into a provider-specific payload.

        Args:
            messages: The input conversation. While often a list of dicts with
                      'role' and 'content' keys, the exact type and structure may vary
                      by implementation.

        Returns:
            A payload in the format expected by the target LLM API. This could be:
            - A list of role-content dictionaries (e.g., for OpenAI)
            - A single formatted string (e.g., for completion-style APIs)
            - A complex dictionary with additional parameters
            - Any other provider-specific data structure
        """
        pass
