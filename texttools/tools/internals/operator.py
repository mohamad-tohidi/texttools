from __future__ import annotations

import math
import re
from typing import Any, TypeVar, Literal, Optional
import json

from openai import OpenAI
from pydantic import BaseModel

from texttools.formatters.user_merge_formatter.user_merge_formatter import (
    UserMergeFormatter,
)
from texttools.tools.internals.prompt_loader import PromptLoader

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
        self.temperature = temperature
        self.client_kwargs = client_kwargs

    def _build_user_message(self, prompt: str) -> dict[str, str]:
        return {"role": "user", "content": prompt}

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

    def _analyze(self, prompt_configs: dict[str, str]) -> str:
        analyze_prompt = prompt_configs["analyze_template"]
        analyze_message = [self._build_user_message(analyze_prompt)]
        analysis = self._analysis_completion(analyze_message)

        return analysis

    def _parse_completion(
        self,
        message: list[dict[str, str]],
        output_model: T,
        logprobs: bool = False,
        top_logprobs: int = 3,
    ) -> tuple[T, Any]:
        try:
            request_kwargs = {
                "model": self.model,
                "messages": message,
                "response_format": output_model,
                "temperature": self.temperature,
                **self.client_kwargs,
            }
            if logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = top_logprobs

            completion = self.client.beta.chat.completions.parse(**request_kwargs)
            parsed = completion.choices[0].message.parsed
            return parsed, completion

        except Exception as e:
            print(f"[ERROR] Failed to parse completion: {e}")
            raise

    def _clean_json_response(self, response: str) -> str:
        """
        Clean JSON response by removing code block markers and whitespace.
        Handles cases like:
        - ```json{"result": "value"}```
        """
        cleaned = response.strip()

        # Remove ```json marker
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]

        # Remove trailing ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    def _convert_to_output_model(self, response_string: str, output_model: T) -> T:
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

            # Fix Python-style booleans
            cleaned_json = cleaned_json.replace("False", "false").replace(
                "True", "true"
            )

            # Convert string to Python dictionary
            response_dict = json.loads(cleaned_json)

            # Convert dictionary to output model
            return output_model(**response_dict)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}\nResponse: {response_string}"
            )
        except Exception as e:
            raise ValueError(f"Failed to convert to output model: {e}")

    def _vllm_completion(
        self,
        message: list[dict[str, str]],
        output_model: T,
        logprobs: bool = False,
        top_logprobs: int = 3,
    ) -> tuple[T, Any]:
        try:
            json_schema = output_model.model_json_schema()

            # Build kwargs dynamically
            request_kwargs = {
                "model": self.model,
                "messages": message,
                "extra_body": {"guided_json": json_schema},
                "temperature": self.temperature,
                **self.client_kwargs,
            }

            if logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = top_logprobs

            completion = self.client.chat.completions.create(**request_kwargs)
            response = completion.choices[0].message.content

            # Convert the string response to output model
            parsed = self._convert_to_output_model(response, output_model)

            return parsed, completion

        except Exception as e:
            print(f"[ERROR] Failed to get vLLM structured output: {e}")
            raise

    def _extract_logprobs(self, completion: dict):
        logprobs_data = []
        ignore_pattern = re.compile(r'^(result|[\s\[\]\{\}",:]+)$')

        for choice in completion.choices:
            if not getattr(choice, "logprobs", None):
                continue

            for logprob_item in choice.logprobs.content:
                if ignore_pattern.match(logprob_item.token):
                    continue
                token_entry = {
                    "token": logprob_item.token,
                    "prob": math.exp(logprob_item.logprob),
                    "top_alternatives": [],
                }
                for alt in logprob_item.top_logprobs:
                    if ignore_pattern.match(alt.token):
                        continue
                    token_entry["top_alternatives"].append(
                        {
                            "token": alt.token,
                            "prob": math.exp(alt.logprob),
                        }
                    )
                logprobs_data.append(token_entry)

        return logprobs_data

    def run(
        self,
        input_text: str,
        prompt_file: str,
        output_model: T,
        with_analysis: bool = False,
        use_modes: bool = False,
        mode: str = "",
        resp_format: Literal["vllm", "parse"] = "parse",
        output_lang: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: int = 3,
        **extra_kwargs,
    ) -> dict[str, Any]:
        """
        Execute the LLM pipeline with the given input text.

        Args:
            input_text: The text to process (will be stripped of whitespace)
            **extra_kwargs: Additional variables to inject into prompt templates

        Returns:
            Dictionary containing the parsed result and optional analysis
        """
        prompt_loader = PromptLoader()
        formatter = UserMergeFormatter()

        try:
            cleaned_text = input_text.strip()

            prompt_configs = prompt_loader.load_prompts(
                prompt_file,
                use_modes,
                mode,
                cleaned_text,
                **extra_kwargs,
            )

            messages: list[dict[str, str]] = []

            if with_analysis:
                analysis = self._analyze(prompt_configs)
                messages.append(
                    self._build_user_message(f"Based on this analysis: {analysis}")
                )

            if output_lang:
                messages.append(
                    self._build_user_message(
                        f"Respond only in the {output_lang} language."
                    )
                )

            messages.append(self._build_user_message(prompt_configs["main_template"]))

            messages = formatter.format(messages)

            if resp_format == "vllm":
                parsed, completion = self._vllm_completion(
                    messages, output_model, logprobs, top_logprobs
                )
            elif resp_format == "parse":
                parsed, completion = self._parse_completion(
                    messages, output_model, logprobs, top_logprobs
                )

            results = {"result": parsed.result}

            if logprobs:
                results["logprobs"] = self._extract_logprobs(completion)

            if with_analysis:
                results["analysis"] = analysis

            return results

        except Exception as e:
            # Print error clearly and exit
            print(f"[ERROR] Operation failed: {e}")
            exit(1)
