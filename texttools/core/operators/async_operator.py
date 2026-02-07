from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from ..exceptions import LLMError, PromptError, TextToolsError, ValidationError
from ..internal_models import OperatorOutput
from ..utils import OperatorUtils


class AsyncOperator:
    """
    Core engine for running text-processing operations with an LLM.
    """

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def _run_analysis(
        self, analysis_messages: list[dict[str, str]]
    ) -> tuple[str, Any]:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=analysis_messages,
            )

            if not completion.choices:
                raise LLMError("No choices returned from LLM")

            analysis = completion.choices[0].message.content.strip()

            if not analysis:
                raise LLMError("Empty analysis response")

            return analysis, completion

        except Exception as e:
            if isinstance(e, (PromptError, LLMError)):
                raise
            raise LLMError(f"Analysis failed: {e}")

    async def _run_completion(
        self,
        main_messages: list[dict[str, str]],
        output_model: type[BaseModel],
        temperature: float,
        logprobs: bool,
        top_logprobs: int,
        priority: int | None,
    ) -> tuple[BaseModel, Any]:
        """
        Parses a chat completion using OpenAI's structured output format.
        Returns both the parsed output and the completion for logprobs.
        """
        try:
            request_kwargs = {
                "model": self._model,
                "messages": main_messages,
                "response_format": output_model,
                "temperature": temperature,
            }

            if logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = top_logprobs

            if priority is not None:
                request_kwargs["extra_body"] = {"priority": priority}

            completion = await self._client.beta.chat.completions.parse(
                **request_kwargs
            )

            if not completion.choices:
                raise LLMError("No choices returned from LLM")

            parsed_output = completion.choices[0].message.parsed

            if not parsed_output:
                raise LLMError("Failed to parse LLM response")

            return parsed_output, completion

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Completion failed: {e}")

    async def run(
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
        output_model: type[BaseModel],
        mode: str | None,
        **extra_kwargs,
    ) -> OperatorOutput:
        """
        Execute the LLM pipeline with the given input text.
        """
        try:
            prompt_configs = OperatorUtils.load_prompt(
                prompt_file=tool_name + ".yaml",
                text=text.strip(),
                mode=mode,
                **extra_kwargs,
            )

            analysis: str | None = None
            analysis_completion: Any = None

            if with_analysis:
                analysis_messages = OperatorUtils.build_message(
                    prompt_configs["analyze_template"]
                )
                analysis, analysis_completion = await self._run_analysis(
                    analysis_messages
                )

            main_prompt = OperatorUtils.build_main_prompt(
                prompt_configs["main_template"], analysis, output_lang, user_prompt
            )
            main_messages = OperatorUtils.build_message(main_prompt)

            parsed_output, main_completion = await self._run_completion(
                main_messages,
                output_model,
                temperature,
                logprobs,
                top_logprobs,
                priority,
            )

            # Retry logic in case output validation fails
            if validator and not validator(parsed_output.result):
                if (
                    not isinstance(max_validation_retries, int)
                    or max_validation_retries < 1
                ):
                    raise ValueError("max_validation_retries should be a positive int")

                succeeded = False
                for _ in range(max_validation_retries):
                    retry_temperature = OperatorUtils.get_retry_temp(temperature)

                    try:
                        parsed_output, main_completion = await self._run_completion(
                            main_messages,
                            output_model,
                            retry_temperature,
                            logprobs,
                            top_logprobs,
                            priority=priority,
                        )

                        # Check if retry was successful
                        if validator(parsed_output.result):
                            succeeded = True
                            break

                    except LLMError:
                        pass

                if not succeeded:
                    raise ValidationError("Validation failed after all retries")

            operator_output = OperatorOutput(
                result=parsed_output.result,
                analysis=analysis if with_analysis else None,
                logprobs=OperatorUtils.extract_logprobs(main_completion)
                if logprobs
                else None,
                processed_by=self._model,
                token_usage=OperatorUtils.extract_token_usage(
                    main_completion, analysis_completion
                ),
            )

            return operator_output

        except (PromptError, LLMError, ValidationError):
            raise
        except Exception as e:
            raise TextToolsError(f"Unexpected error in operator: {e}")
