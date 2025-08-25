from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseFormatter(ABC):
    """
    Given (raw_text, reason, maybe other hints), produce whatever payload.
    A) Single string prompt (for providers that don't support multiple messages) or
    B) List of {role, content} dicts, or
    C) Whatever shape the provider needs.
    """

    @abstractmethod
    def format(
        self,
        text: str,
        reason: Optional[str],
        schema_instr: str,
        prompt_template: Optional[str],
    ) -> Any:
        """
        - For an OpenAI style API, this might return list[{"role": "user"/"assistant", "content": "…"}].
        - For a one shot "text only" API, this might return a single string combining everything.
        - For some niche service, it might return JSON: {"inputs": […], "parameters": {…}}.
        """
        pass
