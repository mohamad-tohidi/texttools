from abc import ABC, abstractmethod
from typing import Any


class BaseFormatter(ABC):
    """
    Given (raw_text, reason, maybe other hints), produce whatever payload.
    1. Single string prompt (for providers that don't support multiple messages)
    2. List of {role, content} dicts
    3. Whatever shape the provider needs
    """

    @abstractmethod
    def format(
        self,
        text: str,
    ) -> Any:
        """
        - For an OpenAI style API, this might return list[{"role": "user"/"assistant", "content": "…"}].
        - For a one shot "text only" API, this might return a single string combining everything.
        - For some niche service, it might return JSON: {"inputs": […], "parameters": {…}}.
        """
        pass
