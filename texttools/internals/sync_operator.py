from typing import TypeVar, Type
from collections.abc import Callable
import logging

from openai import OpenAI
from pydantic import BaseModel

from texttools.internals.models import ToolOutput
from texttools.internals.operator_utils import OperatorUtils
from texttools.internals.prompt_loader import PromptLoader
from texttools.internals.exceptions import (
    TextToolsError,
    LLMError,
    ValidationError,
    PromptError,
)

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("texttools.sync_operator")


class Operator:
    """
    Core engine for running text-processing operations with an LLM (Sync).

    It wires together:
    - `PromptLoader` → loads YAML prompt templates.
    - `UserMergeFormatter` → applies formatting to messages (e.g., merging).
    - OpenAI client → executes completions/parsed completions.
    """

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def _analyze(self, prompt_configs: dict[str, str], temperature: float) -> str:
        """
        Calls OpenAI API for analysis using the configured prompt template.
        Returns the analyzed content as a string.
        """
        try:
            analyze_prompt = prompt_configs["analyze_template"]

            if not analyze_prompt:
                raise PromptError("Analyze template is empty")

            analyze_message = [OperatorUtils.build_user_message(analyze_prompt)]
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=analyze_message,
                temperature=temperature,
            )

            if not completion.choices:
                raise LLMError("No choices returned from LLM")

            analysis = completion.choices[0].message.content.strip()

            if not analysis:
                raise LLMError("Empty analysis response")

            return analysis.strip()

        except Exception as e:
            if isinstance(e, (PromptError, LLMError)):
                raise
            raise LLMError(f"Analysis failed: {e}")

    def _parse_completion(
        self,
        message: list[dict[str, str]],
        output_model: Type[T],
        temperature: float,
        logprobs: bool = False,
        top_logprobs: int = 3,
        priority: int | None = 0,
    ) -> tuple[T, object]:
        """
        Parses a chat completion using OpenAI's structured output format.
        Returns both the parsed object and the raw completion for logprobs.
        """
        try:
            request_kwargs = {
                "model": self._model,
                "messages": message,
                "response_format": output_model,
                "temperature": temperature,
            }

            if logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = top_logprobs
            if priority:
                request_kwargs["extra_body"] = {"priority": priority}
            completion = self._client.beta.chat.completions.parse(**request_kwargs)

            if not completion.choices:
                raise LLMError("No choices returned from LLM")

            parsed = completion.choices[0].message.parsed

            if not parsed:
                raise LLMError("Failed to parse LLM response")

            return parsed, completion

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Completion failed: {e}")

    def run(
        self,
        # User parameters
        text: str,
        with_analysis: bool,
        output_lang: str | None,
        user_prompt: str | None,
        temperature: float,
        logprobs: bool,
        top_logprobs: int | None,
        validator: Callable[[object], bool] | None,
        max_validation_retries: int | None,
        # Internal parameters
        prompt_file: str,
        output_model: Type[T],
        mode: str | None,
        priority: int | None = 0,
        **extra_kwargs,
    ) -> ToolOutput:
        """
        Execute the LLM pipeline with the given input text. (Sync)
        """
        try:
            prompt_loader = PromptLoader()
            output = ToolOutput()

            # Prompt configs contain two keys: main_template and analyze template, both are string
            prompt_configs = prompt_loader.load(
                prompt_file=prompt_file,
                text=text.strip(),
                mode=mode,
                **extra_kwargs,
            )

            messages = []

            if with_analysis:
                analysis = self._analyze(prompt_configs, temperature)
                messages.append(
                    OperatorUtils.build_user_message(
                        f"Based on this analysis: {analysis}"
                    )
                )

            if output_lang:
                messages.append(
                    OperatorUtils.build_user_message(
                        f"Respond only in the {output_lang} language."
                    )
                )

            if user_prompt:
                messages.append(
                    OperatorUtils.build_user_message(
                        f"Consider this instruction {user_prompt}"
                    )
                )

            messages.append(
                OperatorUtils.build_user_message(prompt_configs["main_template"])
            )

            messages = OperatorUtils.user_merge_format(messages)

            if logprobs and (not isinstance(top_logprobs, int) or top_logprobs < 2):
                raise ValueError("top_logprobs should be an integer greater than 1")

            parsed, completion = self._parse_completion(
                messages, output_model, temperature, logprobs, top_logprobs, priority
            )

            output.result = parsed.result

            # Retry logic if validation fails
            if validator and not validator(output.result):
                if (
                    not isinstance(max_validation_retries, int)
                    or max_validation_retries < 1
                ):
                    raise ValueError(
                        "max_validation_retries should be a positive integer"
                    )

                succeeded = False
                for attempt in range(max_validation_retries):
                    logger.warning(
                        f"Validation failed, retrying for the {attempt + 1} time."
                    )

                    # Generate new temperature for retry
                    retry_temperature = OperatorUtils.get_retry_temp(temperature)

                    try:
                        parsed, completion = self._parse_completion(
                            messages,
                            output_model,
                            retry_temperature,
                            logprobs,
                            top_logprobs,
                            priority=priority,
                        )

                        output.result = parsed.result

                        # Check if retry was successful
                        if validator(output.result):
                            succeeded = True
                            break

                    except LLMError as e:
                        logger.error(f"Retry attempt {attempt + 1} failed: {e}")

                if not succeeded:
                    raise ValidationError(
                        f"Validation failed after {max_validation_retries} retries"
                    )

            if logprobs:
                output.logprobs = OperatorUtils.extract_logprobs(completion)

            if with_analysis:
                output.analysis = analysis

            output.process = prompt_file[:-5]

            return output

        except (PromptError, LLMError, ValidationError):
            raise
        except Exception as e:
            raise TextToolsError(f"Unexpected error in operator: {e}")
