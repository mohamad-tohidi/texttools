from __future__ import annotations

from typing import Any, TypeVar, Type, Literal
import json

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
    - RESP_FORMAT: str → "vllm" or "parse"
    """

    PROMPT_FILE: str
    OUTPUT_MODEL: Type[T]
    WITH_ANALYSIS: bool = False
    USE_MODES: bool
    MODE: str = ""
    RESP_FORMAT: Literal["vllm", "parse"] = "vllm"

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

    def _clean_json_response(self, response: str) -> str:
        """
        Clean JSON response by removing code block markers and whitespace.
        Handles cases like:
        - ```json{"result": "value"}```
        - ```{"result": "value"}```
        """
        # Remove code block markers
        cleaned = response.strip()

        # Remove ```json and ``` markers
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]  # Remove ```json
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]  # Remove ```

        # Remove trailing ``` or '''
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    def _convert_to_output_model(self, response_string: str) -> T:
        """
        Convert a JSON response string to output model.

        Args:
            response_string: The JSON string (may contain code block markers)
            output_model: Your Pydantic output model class (e.g., StrOutput, ListStrOutput)

        Returns:
            Instance of your output model
        """
        try:
            # Clean the response string
            cleaned_json = self._clean_json_response(response_string)

            # Convert string to Python dictionary
            response_dict = json.loads(cleaned_json)

            # Convert dictionary to output model
            return self.OUTPUT_MODEL(**response_dict)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}\nResponse: {response_string}"
            )
        except Exception as e:
            raise ValueError(f"Failed to convert to output model: {e}")

    def _vllm_completion(self, message: list[dict[str, str]]) -> T:
        try:
            json_schema = self.OUTPUT_MODEL.model_json_schema()
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                extra_body={"guided_json": json_schema},
                temperature=self.temperature,
                **self.client_kwargs,
            )
            response = completion.choices[0].message.content

            # Convert the string response to output model
            parsed_response = self._convert_to_output_model(response)

            return parsed_response

        except Exception as e:
            print(f"[ERROR] Failed to get vLLM structured output: {e}")
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

            if self.RESP_FORMAT == "vllm":
                parsed = self._vllm_completion(messages)
            elif self.RESP_FORMAT == "parse":
                parsed = self._parse_completion(messages)

            results = {"result": parsed.result}

            if self.WITH_ANALYSIS:
                results["analysis"] = analysis

            return results

        except Exception as e:
            # Print error clearly and exit
            print(f"[ERROR] Operation failed: {e}")
            exit(1)
