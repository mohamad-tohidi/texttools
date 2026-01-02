from collections.abc import Callable
from typing import Any, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from ..engine import OperatorUtils, PromptLoader
from ..exceptions import LLMError, PromptError, TextToolsError, ValidationError
from ..internal_models import OperatorOutput

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)


class Operator:
    """
    Core engine for running text-processing operations with an LLM.
    """

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def _analyze_completion(self, analyze_message: list[dict[str, str]]) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=analyze_message,
            )

            if not completion.choices:
                raise LLMError("No choices returned from LLM")

            analysis = completion.choices[0].message.content.strip()

            if not analysis:
                raise LLMError("Empty analysis response")

            return analysis

        except Exception as e:
            if isinstance(e, (PromptError, LLMError)):
                raise
            raise LLMError(f"Analysis failed: {e}")

    def _parse_completion(
        self,
        main_message: list[dict[str, str]],
        output_model: Type[T],
        temperature: float,
        logprobs: bool,
        top_logprobs: int,
        priority: int | None,
    ) -> tuple[T, Any]:
        """
        Parses a chat completion using OpenAI's structured output format.
        Returns both the parsed and the completion for logprobs.
        """
        try:
            request_kwargs = {
                "model": self._model,
                "messages": main_message,
                "response_format": output_model,
                "temperature": temperature,
            }

            if logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = top_logprobs

            if priority is not None:
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
        text: str,
        with_analysis: bool,
        output_lang: str | None,
        user_prompt: str | None,
        temperature: float,
        logprobs: bool,
        top_logprobs: int,
        validator: Callable[[Any], bool] | None,
        max_validation_retries: int | None,
        priority: int | None,
        tool_name: str,
        output_model: Type[T],
        mode: str | None,
        **extra_kwargs,
    ) -> OperatorOutput:
        """
        Execute the LLM pipeline with the given input text.
        """
        try:
            prompt_loader = PromptLoader()
            prompt_configs = prompt_loader.load(
                prompt_file=tool_name + ".yaml",
                text=text.strip(),
                mode=mode,
                **extra_kwargs,
            )

            analysis: str | None = None

            if with_analysis:
                analyze_message = OperatorUtils.build_message(
                    prompt_configs["analyze_template"]
                )
                analysis = self._analyze_completion(analyze_message)

            main_prompt = OperatorUtils.build_main_prompt(
                prompt_configs["main_template"], analysis, output_lang, user_prompt
            )
            main_message = OperatorUtils.build_message(main_prompt)

            parsed, completion = self._parse_completion(
                main_message,
                output_model,
                temperature,
                logprobs,
                top_logprobs,
                priority,
            )

            # Retry logic in case output validation fails
            if validator and not validator(parsed.result):
                if (
                    not isinstance(max_validation_retries, int)
                    or max_validation_retries < 1
                ):
                    raise ValueError("max_validation_retries should be a positive int")

                succeeded = False
                for _ in range(max_validation_retries):
                    # Generate a new temperature to retry
                    retry_temperature = OperatorUtils.get_retry_temp(temperature)

                    try:
                        parsed, completion = self._parse_completion(
                            main_message,
                            output_model,
                            retry_temperature,
                            logprobs,
                            top_logprobs,
                            priority=priority,
                        )

                        # Check if retry was successful
                        if validator(parsed.result):
                            succeeded = True
                            break

                    except LLMError:
                        pass

                if not succeeded:
                    raise ValidationError("Validation failed after all retries")

            operator_output = OperatorOutput(
                result=parsed.result,
                analysis=analysis if with_analysis else None,
                logprobs=OperatorUtils.extract_logprobs(completion)
                if logprobs
                else None,
            )

            return operator_output

        except (PromptError, LLMError, ValidationError):
            raise
        except Exception as e:
            raise TextToolsError(f"Unexpected error in operator: {e}")
