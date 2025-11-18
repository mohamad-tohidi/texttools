from typing import Any, TypeVar, Type, Callable
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel

from texttools.tools.internals.output_models import ToolOutput
from texttools.tools.internals.operator_utils import OperatorUtils
from texttools.tools.internals.prompt_loader import PromptLoader

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("texttools.async_operator")


class AsyncOperator:
    """
    Core engine for running text-processing operations with an LLM (Async).

    It wires together:
    - `PromptLoader` → loads YAML prompt templates.
    - `UserMergeFormatter` → applies formatting to messages (e.g., merging).
    - AsyncOpenAI client → executes completions/parsed completions.
    """

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def _analyze(self, prompt_configs: dict[str, str], temperature: float) -> str:
        """
        Calls OpenAI API for analysis using the configured prompt template.
        Returns the analyzed content as a string.
        """
        analyze_prompt = prompt_configs["analyze_template"]
        analyze_message = [OperatorUtils.build_user_message(analyze_prompt)]
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=analyze_message,
            temperature=temperature,
        )
        analysis = completion.choices[0].message.content.strip()
        return analysis

    async def _parse_completion(
        self,
        message: list[dict[str, str]],
        output_model: Type[T],
        temperature: float,
        logprobs: bool = False,
        top_logprobs: int = 3,
    ) -> tuple[T, Any]:
        """
        Parses a chat completion using OpenAI's structured output format.
        Returns both the parsed object and the raw completion for logprobs.
        """
        request_kwargs = {
            "model": self._model,
            "messages": message,
            "response_format": output_model,
            "temperature": temperature,
        }

        if logprobs:
            request_kwargs["logprobs"] = True
            request_kwargs["top_logprobs"] = top_logprobs

        completion = await self._client.beta.chat.completions.parse(**request_kwargs)
        parsed = completion.choices[0].message.parsed
        return parsed, completion

    async def run(
        self,
        # User parameters
        text: str,
        with_analysis: bool,
        output_lang: str | None,
        user_prompt: str | None,
        temperature: float,
        logprobs: bool,
        top_logprobs: int | None,
        validator: Callable[[Any], bool] | None,
        max_validation_retries: int | None,
        # Internal parameters
        prompt_file: str,
        output_model: Type[T],
        mode: str | None,
        **extra_kwargs,
    ) -> ToolOutput:
        """
        Execute the async LLM pipeline with the given input text. (Async)
        """
        prompt_loader = PromptLoader()
        output = ToolOutput()

        try:
            # Prompt configs contain two keys: main_template and analyze template, both are string
            prompt_configs = prompt_loader.load(
                prompt_file=prompt_file,
                text=text.strip(),
                mode=mode,
                **extra_kwargs,
            )

            messages = []

            if with_analysis:
                analysis = await self._analyze(prompt_configs, temperature)
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

            parsed, completion = await self._parse_completion(
                messages, output_model, temperature, logprobs, top_logprobs
            )

            # Ensure output_model has a `result` field
            if not hasattr(parsed, "result"):
                error = "The provided output_model must define a field named 'result'"
                logger.error(error)
                output.errors.append(error)
                return output

            output.result = parsed.result

            # Retry logic if validation fails
            if validator and not validator(output.result):
                for attempt in range(max_validation_retries):
                    logger.warning(
                        f"Validation failed, retrying for the {attempt + 1} time."
                    )

                    # Generate new temperature for retry
                    retry_temperature = OperatorUtils.get_retry_temp(temperature)
                    try:
                        parsed, completion = await self._parse_completion(
                            messages,
                            output_model,
                            retry_temperature,
                            logprobs,
                            top_logprobs,
                        )

                        output.result = parsed.result

                        # Check if retry was successful
                        if validator(output.result):
                            logger.info(
                                f"Validation passed on retry attempt {attempt + 1}"
                            )
                            break
                        else:
                            logger.warning(
                                f"Validation still failing after retry attempt {attempt + 1}"
                            )

                    except Exception as e:
                        logger.error(f"Retry attempt {attempt + 1} failed: {e}")
                        # Continue to next retry attempt if this one fails

            # Final check after all retries
            if validator and not validator(output.result):
                output.errors.append("Validation failed after all retry attempts")

            if logprobs:
                output.logprobs = OperatorUtils.extract_logprobs(completion)

            if with_analysis:
                output.analysis = analysis

            return output

        except Exception as e:
            logger.error(f"AsyncTheTool failed: {e}")
            output.errors.append(str(e))
            return output
