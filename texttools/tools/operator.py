from __future__ import annotations

from typing import Any, TypeVar, Type

from openai import OpenAI
from pydantic import BaseModel

from texttools.formatters.user_merge_formatter.user_merge_formatter import (
    UserMergeFormatter,
)
from texttools.tools.prompt_loader import PromptLoader

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)


class Operator:
    """
    Core engine for running text-processing operations with an LLM.

    It wires together:
    - `PromptLoader` → loads YAML prompt templates.
    - `UserMergeFormatter` → applies formatting to messages (e.g., merging).
    - OpenAI client → executes completions/parsed completions.

    Workflow inside `run()`:
    1. Load prompt templates (`main_template` [+ `analyze_template` if enabled]).
    2. Optionally generate an "analysis" step via `_analyze()`.
    3. Build messages for the LLM.
    4. Call `.beta.chat.completions.parse()` to parse the result into the
       configured `OUTPUT_MODEL` (a Pydantic schema).
    5. Return results as a dict (always `{"result": ...}`, plus `analysis`
       if analysis was enabled).

    Attributes configured dynamically by `TheTool`:
    - PROMPT_FILE: str → YAML filename
    - OUTPUT_MODEL: Pydantic model class
    - WITH_ANALYSIS: bool → whether to run an analysis phase first
    - USE_MODES: bool → whether to select prompts by mode
    - MODE: str → which mode to use if modes are enabled
    """

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
        temperature: float = 0.0,
        **client_kwargs: Any,
    ):
        self.client: OpenAI = client
        self.model = model
        self.prompt_loader = PromptLoader()
        self.formatter = UserMergeFormatter()
        self.temperature = temperature
        self.client_kwargs = client_kwargs

    def _build_user_message(self, prompt: str) -> dict[str, str]:
        return {"role": "user", "content": prompt}

    def _apply_formatter(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return self.formatter.format(messages)

    def _analysis_completion(self, analyze_message: list[dict[str, str]]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=analyze_message,
                temperature=self.temperature,
                **self.client_kwargs,
            )
            analysis = completion.choices[0].message.content.strip()
            return analysis

        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            raise

    def _analyze(self) -> str:
        analyze_prompt = self.prompt_configs["analyze_template"]
        analyze_message = [self._build_user_message(analyze_prompt)]
        analysis = self._analysis_completion(analyze_message)

        return analysis

    def _build_main_message(self) -> list[dict[str, str]]:
        main_prompt = self.prompt_configs["main_template"]
        main_message = self._build_user_message(main_prompt)

        return main_message

    def _parse_completion(self, message: list[dict[str, str]]) -> T:
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=message,
                response_format=self.OUTPUT_MODEL,
                temperature=self.temperature,
                **self.client_kwargs,
            )
            parsed = completion.choices[0].message.parsed
            return parsed

        except Exception as e:
            print(f"[ERROR] Failed to parse completion: {e}")
            raise

    def run(self, input_text: str, **extra_kwargs) -> dict[str, Any]:
        """
        Execute the LLM pipeline with the given input text.

        Args:
            input_text: The text to process (will be stripped of whitespace)
            **extra_kwargs: Additional variables to inject into prompt templates

        Returns:
            Dictionary containing the parsed result and optional analysis
        """
        try:
            cleaned_text = input_text.strip()

            self.prompt_configs = self.prompt_loader.load_prompts(
                self.PROMPT_FILE,
                self.USE_MODES,
                self.MODE,
                cleaned_text,
                **extra_kwargs,
            )

            messages: list[dict[str, str]] = []

            if self.WITH_ANALYSIS:
                analysis = self._analyze()
                messages.append(
                    self._build_user_message(f"Based on this analysis: {analysis}")
                )

            messages.append(self._build_main_message())
            messages = self.formatter.format(messages)
            parsed = self._parse_completion(messages)
            results = {"result": parsed.result}

            if self.WITH_ANALYSIS:
                results["analysis"] = analysis

            return results

        except Exception as e:
            # Print error clearly and exit
            print(f"[ERROR] Operation failed: {e}")
            exit(1)
