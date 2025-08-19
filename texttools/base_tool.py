from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import yaml
from openai import OpenAI
from pydantic import BaseModel

from texttools.formatter import Formatter

T = TypeVar("T", bound=BaseModel)


class BaseTool:
    """
    Base class:
    - Loads YAML with both `main_template` and `reason_template`
    - Runs optional reasoning step before main task
    - Returns parsed Pydantic model
    """

    # These will be set by subclass
    prompt_file: str = ""
    output_model: Type[T]

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        prompts_dir: str = "prompts",
        chat_formatter=Formatter(),
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
        if not self.main_template:
            raise ValueError(f"Missing `main_template` in {self.prompt_file}")
        self.reason_template = data.get("reason_template")

    def _apply_formatter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        formatted = self.chat_formatter.format(messages=messages)
        return formatted

    def _dispatch(self, results: dict) -> None:
        for handler in self.handlers:
            try:
                handler.handle(results)
            except Exception:
                pass

    def _prompt_to_dict(self, prompt: str):
        return [{"role": "user", "content": prompt}]

    def _build_messages(
        self, input_text: str, reason: Optional[str]
    ) -> list[dict[str, str]]:
        prompt = self.main_template.format(input=input_text, reason=(reason or ""))
        messages = self._prompt_to_dict(prompt)

        if self.chat_formatter:
            formatted = self._apply_formatter(messages)

        return formatted

    def _reason(self, input_text: str) -> str:
        """
        Default reasoning step: uses `reason_template` if available.
        """
        prompt = self.reason_template.format(input=input_text)
        messages = messages = self._prompt_to_dict(prompt)
        formatted = self._apply_formatter(messages)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        reason_text = completion.choices[0].message.content.strip()

        return reason_text

    def run(self, input_text: str) -> T:
        """
        Run the tool:
        - Optionally runs reasoning (if enabled).
        - Fills main template with input and reasoning.
        - Calls model and parses output into `self.output_model`.
        """
        cleaned_text = input_text.strip()

        if self.use_reason and self.reason_template:
            reason_text = self._reason(cleaned_text)
        else:
            reason_text = None

        messages = self._build_messages(cleaned_text, reason_text)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=self.output_model,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        parsed = completion.choices[0].message.parsed

        return parsed
