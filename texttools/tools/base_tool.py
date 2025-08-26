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
    """
    Loads YAML with both `main_template` and `reason_template`
    """

    prompt_dir_name: str = "prompts"

    def _load_templates(
        self, prompt_file_name: str, use_modes: bool, mode: str
    ) -> dict[str, str]:
        # prompt_file_name has the .yaml suffix, so to access the tool folder name, .yaml suffix should be removed
        tool_name = prompt_file_name.removesuffix(".yaml")
        prompt_file = (
            Path(__file__).parent.parent
            / self.prompt_dir_name
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

    def _build_format_args(self, input_text: str, **extra_kwargs) -> dict[str, str]:
        # Base formatting args
        format_args = {"input": input_text}
        # Merge extras
        format_args.update(extra_kwargs)
        return format_args

    def load_prompts(
        self,
        prompt_file_name: str,
        use_modes: bool,
        mode: str,
        input_text: str,
        **extra_kwargs,
    ) -> dict[str, str]:
        template_configs = self._load_templates(prompt_file_name, use_modes, mode)
        format_args = self._build_format_args(input_text, **extra_kwargs)

        for key in template_configs.keys():
            template_configs[key] = template_configs[key].format(**format_args)

        return template_configs


class BaseTool:
    """
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
        self.prompt_loader = prompt_loader
        self.formatter = formatter
        self.use_reason = use_reason
        self.temperature = temperature
        self.handlers = handlers or []
        self.client_kwargs = client_kwargs

    def _apply_formatter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        formatted = self.formatter.format(messages=messages)
        return formatted

    def _prompt_to_dict(self, prompt: str) -> dict[str, str]:
        return {"role": "user", "content": prompt}

    def _result_to_dict(self, input_text: str, result: Any) -> dict[str, Any]:
        return {"input_text": input_text, "result": result}

    def _reason(self, prompt_configs: dict[str, str]) -> str:
        reason_prompt = prompt_configs["reason_template"]

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

    def _build_messages(self, prompt_configs: dict[str, str]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if self.use_reason and prompt_configs["reason_template"]:
            reason = self._reason(prompt_configs)
            messages.append(self._prompt_to_dict(f"Based on this analysis: {reason}"))

        main_prompt = prompt_configs["main_template"]

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

        prompt_configs = self.prompt_loader.load_prompts(
            self.prompt_file, self.use_modes, self.mode, cleaned_text, **extra_kwargs
        )

        messages = self._build_messages(prompt_configs)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=self.output_model,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        parsed = completion.choices[0].message.parsed

        return parsed
