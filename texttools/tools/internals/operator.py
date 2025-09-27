from __future__ import annotations

import math
import re
from typing import Any, TypeVar, Type, Literal, Optional
import json
import logging

from openai import OpenAI
from pydantic import BaseModel

from texttools.formatters.user_merge_formatter.user_merge_formatter import (
    UserMergeFormatter,
)
from texttools.tools.internals.prompt_loader import PromptLoader

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)

# Configure logger
logger = logging.getLogger("operator")
logger.setLevel(logging.INFO)


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

    def __init__(self, client: OpenAI):
        self.client: OpenAI = client

    def _build_user_message(self, prompt: str) -> dict[str, str]:
        return {"role": "user", "content": prompt}

    def _analysis_completion(
        self,
        analyze_message: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=model,
            messages=analyze_message,
            temperature=temperature,
        )
        analysis = completion.choices[0].message.content.strip()
        return analysis

    def _analyze(
        self,
        prompt_configs: dict[str, str],
        model: str,
        temperature: float,
    ) -> str:
        analyze_prompt = prompt_configs["analyze_template"]
        analyze_message = [self._build_user_message(analyze_prompt)]
        analysis = self._analysis_completion(analyze_message, model, temperature)
        return analysis

    def _parse_completion(
        self,
        message: list[dict[str, str]],
        output_model: Type[T],
        model: str,
        temperature: float,
        logprobs: bool = False,
        top_logprobs: int = 3,
    ) -> tuple[Type[T], Any]:
        request_kwargs = {
            "model": model,
            "messages": message,
            "response_format": output_model,
            "temperature": temperature,
        }
        if logprobs:
            request_kwargs["logprobs"] = True
            request_kwargs["top_logprobs"] = top_logprobs

        completion = self.client.beta.chat.completions.parse(**request_kwargs)
        parsed = completion.choices[0].message.parsed
        return parsed, completion

    def _clean_json_response(self, response: str) -> str:
        """
        Clean JSON response by removing code block markers and whitespace.
        Handles cases like:
        - ```json{"result": "value"}```
        """
        stripped = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", stripped)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        return cleaned.strip()

    def _convert_to_output_model(
        self, response_string: str, output_model: Type[T]
    ) -> Type[T]:
        """
        Convert a JSON response string to output model.

        Args:
            response_string: The JSON string (may contain code block markers)
            output_model: Your Pydantic output model class (e.g., StrOutput, ListStrOutput)

        Returns:
            Instance of your output model
        """
        # Clean the response string
        cleaned_json = self._clean_json_response(response_string)

        # Fix Python-style booleans
        cleaned_json = cleaned_json.replace("False", "false").replace("True", "true")

        # Convert string to Python dictionary
        response_dict = json.loads(cleaned_json)

        # Convert dictionary to output model
        return output_model(**response_dict)

    def _vllm_completion(
        self,
        message: list[dict[str, str]],
        output_model: Type[T],
        model: str,
        temperature: float,
        logprobs: bool = False,
        top_logprobs: int = 3,
    ) -> tuple[Type[T], Any]:
        json_schema = output_model.model_json_schema()

        # Build kwargs dynamically
        request_kwargs = {
            "model": model,
            "messages": message,
            "extra_body": {"guided_json": json_schema},
            "temperature": temperature,
        }

        if logprobs:
            request_kwargs["logprobs"] = True
            request_kwargs["top_logprobs"] = top_logprobs

        completion = self.client.chat.completions.create(**request_kwargs)
        response = completion.choices[0].message.content

        # Convert the string response to output model
        parsed = self._convert_to_output_model(response, output_model)
        return parsed, completion

    def _extract_logprobs(self, completion: dict):
        logprobs_data = []
        ignore_pattern = re.compile(r'^(result|[\s\[\]\{\}",:]+)$')

        for choice in completion.choices:
            if not getattr(choice, "logprobs", None):
                logger.info("No logprobs found.")
                continue

            for logprob_item in choice.logprobs.content:
                if ignore_pattern.match(logprob_item.token):
                    continue
                token_entry = {
                    "token": logprob_item.token,
                    "prob": round(math.exp(logprob_item.logprob), 8),
                    "top_alternatives": [],
                }
                for alt in logprob_item.top_logprobs:
                    if ignore_pattern.match(alt.token):
                        continue
                    token_entry["top_alternatives"].append(
                        {
                            "token": alt.token,
                            "prob": round(math.exp(alt.logprob), 8),
                        }
                    )
                logprobs_data.append(token_entry)

        return logprobs_data

    def run(
        self,
        text: str,
        # User parameters
        model: str,
        with_analysis: bool,
        temperature: float,
        logprobs: bool,
        top_logprobs: int,
        user_prompt: Optional[str],
        output_lang: Optional[str],
        # Each tool's parameters
        prompt_file: str,
        output_model: Type[T],
        resp_format: Literal["vllm", "parse"] = "parse",
        mode: Optional[str] = None,
        **extra_kwargs,
    ) -> dict[str, Any]:
        """
        Execute the LLM pipeline with the given input text.

        Args:
            text: The text to process (will be stripped of whitespace)
            **extra_kwargs: Additional variables to inject into prompt templates

        Returns:
            Dictionary containing the parsed result and optional analysis
        """
        prompt_loader = PromptLoader()
        formatter = UserMergeFormatter()

        try:
            cleaned_text = text.strip()

            prompt_configs = prompt_loader.load_prompts(
                prompt_file=prompt_file,
                text=cleaned_text,
                mode=mode,
                **extra_kwargs,
            )

            messages: list[dict[str, str]] = []

            if with_analysis:
                analysis = self._analyze(prompt_configs, model, temperature)
                messages.append(
                    self._build_user_message(f"Based on this analysis: {analysis}")
                )

            if output_lang:
                messages.append(
                    self._build_user_message(
                        f"Respond only in the {output_lang} language."
                    )
                )

            if user_prompt:
                messages.append(
                    self._build_user_message(f"Consider this instruction {user_prompt}")
                )

            messages.append(self._build_user_message(prompt_configs["main_template"]))

            messages = formatter.format(messages)

            if resp_format == "vllm":
                parsed, completion = self._vllm_completion(
                    messages, output_model, model, temperature, logprobs, top_logprobs
                )
            elif resp_format == "parse":
                parsed, completion = self._parse_completion(
                    messages, output_model, model, temperature, logprobs, top_logprobs
                )

            # Ensure output_model has a `result` field
            if not hasattr(parsed, "result"):
                logger.error(
                    "The provided output_model must define a field named 'result'"
                )

            results = {"result": parsed.result}

            if logprobs:
                results["logprobs"] = self._extract_logprobs(completion)

            if with_analysis:
                results["analysis"] = analysis

            return results

        except Exception as e:
            logger.error(f"Operation failed: {e}")
            return {"Error": str(e), "result": ""}
