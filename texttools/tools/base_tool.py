from __future__ import annotations

from pathlib import Path
from typing import Any, Type, TypeVar

import yaml
from openai import OpenAI
from pydantic import BaseModel

from texttools.formatters.user_merge_formatter.user_merge_formatter import (
    UserMergeFormatter,
)

T = TypeVar("T", bound=BaseModel)

MAIN_TEMPLATE = "main_template"
ANALYZE_TEMPLATE = "analyze_template"


class PromptLoader:
    """
    Loads YAML with both `main_template` and `analyze_template`
    """

    prompt_dir: str = "prompts"

    def _load_templates(
        self, prompt_file: str, use_modes: bool, mode: str
    ) -> dict[str, str]:
        prompt_file = Path(__file__).parent.parent / self.prompt_dir / prompt_file

        data = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))

        return {
            MAIN_TEMPLATE: data[MAIN_TEMPLATE][mode]
            if use_modes
            else data[MAIN_TEMPLATE],
            ANALYZE_TEMPLATE: data.get(ANALYZE_TEMPLATE)[mode]
            if use_modes
            else data.get(ANALYZE_TEMPLATE),
        }

    def _build_format_args(self, input_text: str, **extra_kwargs) -> dict[str, str]:
        # Base formatting args
        format_args = {"input": input_text}
        # Merge extras
        format_args.update(extra_kwargs)
        return format_args

    def load_prompts(
        self,
        prompt_file: str,
        use_modes: bool,
        mode: str,
        input_text: str,
        **extra_kwargs,
    ) -> dict[str, str]:
        template_configs = self._load_templates(prompt_file, use_modes, mode)
        format_args = self._build_format_args(input_text, **extra_kwargs)

        for key in template_configs.keys():
            template_configs[key] = template_configs[key].format(**format_args)

        return template_configs


class BaseTool:
    """
    - Runs optional analyzing step before main task
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
        with_analysis: bool = False,
        temperature: float = 0.0,
        **client_kwargs: Any,
    ):
        self.client: OpenAI = client
        self.model = model
        self.mode = mode
        self.prompt_loader = prompt_loader
        self.formatter = formatter
        self.with_analysis = with_analysis
        self.temperature = temperature
        self.client_kwargs = client_kwargs

    def _apply_formatter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        formatted = self.formatter.format(messages=messages)
        return formatted

    def _prompt_to_dict(self, prompt: str) -> dict[str, str]:
        return {"role": "user", "content": prompt}

    def _result_to_dict(self, result: Any) -> dict[str, Any]:
        return {"result": result}

    def _analysis_completion(self, messages: list[dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        analyze = completion.choices[0].message.content.strip()

        return analyze

    def _analyze(self) -> str:
        analyze_prompt = self.prompt_configs[ANALYZE_TEMPLATE]
        messages = [self._prompt_to_dict(analyze_prompt)]
        formatted_messages = self._apply_formatter(messages)

        analyze = self._analysis_completion(formatted_messages)

        return analyze

    def _build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if self.with_analysis and self.prompt_configs[ANALYZE_TEMPLATE]:
            analysis = self._analyze()
            messages.append(self._prompt_to_dict(f"Based on this analysis: {analysis}"))

        main_prompt = self.prompt_configs[MAIN_TEMPLATE]
        messages.append(self._prompt_to_dict(main_prompt))
        formatted_messages = self._apply_formatter(messages)

        return formatted_messages

    def _parse(self, messages: list[dict[str, str]]) -> T:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=self.output_model,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        parsed = completion.choices[0].message.parsed

        return parsed

    def run(self, input_text: str, **extra_kwargs) -> T:
        """
        Run the tool:
        - Optionally runs analyzeing (if enabled).
        - Fills main template with input and analyzeing.
        - Calls model and parses output into `self.output_model`.
        """
        cleaned_text = input_text.strip()

        self.prompt_configs = self.prompt_loader.load_prompts(
            self.prompt_file, self.use_modes, self.mode, cleaned_text, **extra_kwargs
        )

        messages = self._build_messages()
        parsed = self._parse(messages)

        return parsed
