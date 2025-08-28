from __future__ import annotations

from typing import Any, TypeVar, Type, Literal

from openai import OpenAI
from pydantic import BaseModel

from texttools.formatters.user_merge_formatter.user_merge_formatter import (
    UserMergeFormatter,
)
from texttools.tools.prompt_loader import PromptLoader
import texttools.tools.output_models as OutputModels

T = TypeVar("T", bound=BaseModel)


class TheTool:
    PROMPT_FILE: str
    OUTPUT_MODEL: Type[T]
    WITH_ANALYSIS: bool = False
    USE_MODES: bool
    MODE: str = ""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        prompt_loader=PromptLoader(),
        formatter=UserMergeFormatter(),
        temperature: float = 0.0,
        **client_kwargs: Any,
    ):
        self.client: OpenAI = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.formatter = formatter
        self.temperature = temperature
        self.client_kwargs = client_kwargs

    def _build_user_message(self, prompt: str) -> dict[str, str]:
        return {"role": "user", "content": prompt}

    def _build_results_dict(self, result: Any) -> dict[str, Any]:
        return {"result": result}

    def _apply_formatter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.formatter.format(messages)

    def _analysis_completion(self, analyze_message: list[dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=analyze_message,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        analysis = completion.choices[0].message.content.strip()

        return analysis

    def _analyze(self) -> str:
        analyze_prompt = self.prompt_configs["analyze_template"]
        analyze_message = [self._build_user_message(analyze_prompt)]
        analysis = self._analysis_completion(analyze_message)

        return analysis

    def _build_main_message(self) -> list[dict[str, str]]:
        main_prompt = self.prompt_configs["main_template"]
        main_message = self._build_user_message(main_prompt)

        return main_message

    def _parse(self, message: list[dict[str, str]]) -> T:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=message,
            response_format=self.OUTPUT_MODEL,
            temperature=self.temperature,
            **self.client_kwargs,
        )
        parsed = completion.choices[0].message.parsed

        return parsed

    def _run(self, input_text: str, **extra_kwargs) -> dict[str, Any]:
        cleaned_text = input_text.strip()

        self.prompt_configs = self.prompt_loader.load_prompts(
            self.PROMPT_FILE, self.USE_MODES, self.MODE, cleaned_text, **extra_kwargs
        )

        messages: list[dict[str, str]] = []

        if self.WITH_ANALYSIS:
            analysis = self._analyze()
            messages.append(
                self._build_user_message(f"Based on this analysis: {analysis}")
            )

        messages.append(self._build_main_message())
        messages = self.formatter.format(messages)
        parsed = self._parse(messages)
        results = {"result": parsed.result}

        if self.WITH_ANALYSIS:
            results["analysis"] = analysis

        return results

    def categorize(self, text: str, with_analysis: bool = False) -> dict[str, str]:
        """
        Text categorizer for Islamic studies domain with optional reasoning step.
        Uses an LLM prompt (`categorizer.yaml`) to assign a single `main_tag`
        from a fixed set of categories (e.g., "باورهای دینی", "اخلاق اسلامی", ...).
        Outputs JSON with one field: {"main_tag": "..."}.
        """
        self.PROMPT_FILE = "categorizer.yaml"
        self.OUTPUT_MODEL = OutputModels.CategorizerOutput
        self.WITH_ANALYSIS = with_analysis
        self.USE_MODES = False

        results = self._run(text)
        return results

    def merge_questions(
        self,
        questions: list[str],
        mode: Literal["default_mode", "reason_mode"] = "default_mode",
        with_analysis: bool = False,
    ) -> dict[str, str]:
        """
        Questions merger with optional reasoning step and two modes:
        1. Default mode
        2. Reason mode
        Outputs JSON with one field: {"merged_question": "..."}.
        """
        question_str = ", ".join(questions)

        self.PROMPT_FILE = "question_merger.yaml"
        self.OUTPUT_MODEL = OutputModels.StrOutput
        self.WITH_ANALYSIS = with_analysis
        self.USE_MODES = True
        self.MODE = mode

        results = self._run(question_str)
        return results
