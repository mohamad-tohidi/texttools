from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseTool:
    """
    Base class:
    - Loads YAML with both `main_template` and `reason_template`
    - Runs optional reasoning step before main task
    - Returns parsed Pydantic model
    """

    # These will be set by subclass
    PROMPT_FILE: str = ""
    OUTPUT_MODEL: Type[T]

    def __init__(
        self,
        client,
        *,
        model: str,
        prompts_dir: str = "prompts",
        chat_formatter: Optional[Any] = None,
        use_reason: bool = False,
        temperature: float = 0.0,
        handlers: Optional[list[Any]] = None,
        **client_kwargs: Any,
    ):
        self.client: OpenAI = client
        self.model = model
        self.prompts_dir = Path(prompts_dir)
        self.chat_formatter = chat_formatter
        self.use_reason = use_reason
        self.temperature = temperature
        self.handlers = handlers or []
        self.client_kwargs = client_kwargs
        data = yaml.safe_load(
            (self.prompts_dir / self.prompt_file).read_text(encoding="utf-8")
        )
        self.main_template = data["main_template"]
        self.reason_template = data.get("reason_template")

    def _apply_formatter(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if self.chat_formatter is None:
            return messages
        return self.chat_formatter.format(messages=messages)

    def _dispatch(self, results: dict) -> None:
        for handler in self.handlers:
            try:
                handler.handle(results)
            except Exception:
                pass

    # Reasoning step
    def reason(self, input_text: str) -> str:
        """
        Default reasoning step: uses `reason_template` if available.
        """
        if not self.reason_template:
            return ""
        prompt = self.reason_template.format(input=input_text)
        messages = [{"role": "user", "content": prompt}]
        messages = self._apply_formatter(messages)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        return resp.choices[0].message.content.strip()

    # Main task
    def _build_messages(
        self, input_text: str, reason: Optional[str]
    ) -> List[Dict[str, str]]:
        prompt = self.main_template.format(input=input_text, reason=(reason or ""))
        msgs = [{"role": "user", "content": prompt}]
        return self._apply_formatter(msgs)

    def run(self, input_text: str) -> T:
        clean = input_text.strip()
        r = self.reason(clean) if self.use_reason else None
        messages = self._build_messages(clean, r)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=self.output_model,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        return completion.choices[0].message.parsed
