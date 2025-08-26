from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import yaml
from openai import OpenAI
from pydantic import BaseModel

from texttools.formatter import UserMergeFormatter

T = TypeVar("T", bound=BaseModel)


class BaseTool:
    """
    - Loads YAML with both `main_template` and `reason_template`
    - Runs optional reasoning step before main task
    - Returns parsed Pydantic model
    """

    # This is the name of tool + .yaml as we need to load the prompt file.
    prompt_file: str = ""

    output_model: Type[T]
    use_modes: bool = False

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        mode: str = "",
        prompts_dir: str = "prompts",
        chat_formatter=UserMergeFormatter(),
        use_reason: bool = False,
        temperature: float = 0.0,
        handlers: Optional[list[Any]] = None,
        **client_kwargs: Any,
    ):
        self.client: OpenAI = client
        self.model = model
        self.mode = mode
        self.prompts_dir = (
            Path(__file__).parent / prompts_dir / self.prompt_file.removesuffix(".yaml")
        )
        self.chat_formatter = chat_formatter
        self.use_reason = use_reason
        self.temperature = temperature
        self.handlers = handlers or []
        self.client_kwargs = client_kwargs
        data = yaml.safe_load(
            (self.prompts_dir / self.prompt_file).read_text(encoding="utf-8")
        )

        if self.use_modes:
            key = self.mode
            self.main_template = data["main_template"][key]
            self.reason_template = data.get("reason_template")[key]
        else:
            self.main_template = data["main_template"]
            self.reason_template = data.get("reason_template")

    def _apply_formatter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        formatted = self.chat_formatter.format(messages=messages)
        return formatted

    def _prompt_to_dict(self, prompt: str):
        return [{"role": "user", "content": prompt}]

    def _build_messages(
        self, input_text: str, **extra_kwargs: Any
    ) -> list[dict[str, str]]:
        # Base formatting args
        format_args = {
            "input": input_text,
        }
        # Merge extras
        format_args.update(extra_kwargs)

        main_prompt = self.main_template.format(**format_args)
        messages = self._prompt_to_dict(main_prompt)

        if self.use_reason and self.reason_template:
            reason = self._reason(input_text, **extra_kwargs)
            messages.append(
                {"role": "user", "content": f"Based on this analysis: {reason}"}
            )

        formatted_messages = self._apply_formatter(messages)

        return formatted_messages

    def _reason(self, input_text: str, **extra_kwargs: Any) -> str:
        # Base formatting args
        format_args = {
            "input": input_text,
        }
        # Merge extras
        format_args.update(extra_kwargs)

        reason_prompt = self.reason_template.format(**format_args)
        messages = self._prompt_to_dict(reason_prompt)
        formatted_messages = self._apply_formatter(messages)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        reason = completion.choices[0].message.content.strip()

        return reason

    def _dispatch(self, results: dict[str, Any]) -> None:
        for handler in self.handlers:
            try:
                handler.handle(results)
            except Exception:
                pass

    def run(self, input_text: str, **kwargs) -> T:
        """
        Run the tool:
        - Optionally runs reasoning (if enabled).
        - Fills main template with input and reasoning.
        - Calls model and parses output into `self.output_model`.
        """
        cleaned_text = input_text.strip()

        messages = self._build_messages(cleaned_text, **kwargs)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=self.output_model,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        parsed = completion.choices[0].message.parsed

        return parsed
