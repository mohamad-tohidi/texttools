from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import yaml
from openai import OpenAI
from pydantic import BaseModel

from texttools.formatters.user_merge_formatter.user_merge_formatter import (
    UserMergeFormatter,
)

T = TypeVar("T", bound=BaseModel)


class PromptLoader:
    PROMPT_DIR_NAME: str = "prompts"

    def load_prompts(
        self, prompt_file_name: str, use_modes: bool, mode: str
    ) -> dict[str, str]:
        tool_name = prompt_file_name.removesuffix(".yaml")
        prompt_file = (
            Path(__file__).parent.parent
            / self.PROMPT_DIR_NAME
            / tool_name
            / prompt_file_name
        )

        data = yaml.safe_load(prompt_file.read_text())
        return {
            "main_template": data["main_template"][mode]
            if use_modes
            else data["main_template"],
            "reason_template": data.get("reason_template")[mode]
            if use_modes
            else data.get("reason_template"),
        }


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
        prompt_loader=PromptLoader(),
        formatter=UserMergeFormatter(),
        use_reason: bool = False,
        temperature: float = 0.0,
        handlers: Optional[list[Any]] = None,
        **client_kwargs: Any,
    ):
        self.client: OpenAI = client
        self.model = model
        self.mode = mode
        self.formatter = formatter
        self.use_reason = use_reason
        self.temperature = temperature
        self.handlers = handlers or []
        self.client_kwargs = client_kwargs

        # Load prompts
        self.prompt_configs = prompt_loader.load_prompts(
            self.prompt_file, self.use_modes, self.mode
        )

    def _apply_formatter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        formatted = self.formatter.format(messages=messages)
        return formatted

    def _prompt_to_dict(self, prompt: str) -> dict[str, str]:
        return {"role": "user", "content": prompt}

    def _result_to_dict(self, input_text: str, result: Any) -> dict[str, Any]:
        return {"input_text": input_text, "result": result}

    def _build_format_args(self, input_text: str, **extra_kwargs) -> dict[str, str]:
        # Base formatting args
        format_args = {"input": input_text}
        # Merge extras
        format_args.update(extra_kwargs)
        return format_args

    def _reason(self, input_text: str, **extra_kwargs: Any) -> str:
        reason_template = self.prompt_configs["reason_template"]
        format_args = self._build_format_args(input_text, **extra_kwargs)
        reason_prompt = reason_template.format(**format_args)
        messages: list[dict[str, str]] = []
        messages.append(self._prompt_to_dict(reason_prompt))
        formatted_messages = self._apply_formatter(messages)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        reason = completion.choices[0].message.content.strip()

        return reason

    def _build_messages(
        self, input_text: str, **extra_kwargs: Any
    ) -> list[dict[str, str]]:
        format_args = self._build_format_args(input_text, **extra_kwargs)

        messages: list[dict[str, str]] = []

        if self.use_reason and self.prompt_configs["reason_template"]:
            reason = self._reason(input_text, **extra_kwargs)
            messages.append(self._prompt_to_dict(f"Based on this analysis: {reason}"))

        main_template = self.prompt_configs["main_template"]
        main_prompt = main_template.format(**format_args)
        messages.append(self._prompt_to_dict(main_prompt))

        formatted_messages = self._apply_formatter(messages)

        return formatted_messages

    def _dispatch(self, results: dict[str, Any]) -> None:
        for handler in self.handlers:
            try:
                handler.handle(results)
            except Exception:
                pass

    def run(self, input_text: str, **extra_kwargs) -> T:
        """
        Run the tool:
        - Optionally runs reasoning (if enabled).
        - Fills main template with input and reasoning.
        - Calls model and parses output into `self.output_model`.
        """
        cleaned_text = input_text.strip()

        messages = self._build_messages(cleaned_text, **extra_kwargs)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=self.output_model,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        parsed = completion.choices[0].message.parsed

        return parsed
