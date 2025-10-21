from typing import Any, TypeVar, Type, Literal
import logging

from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.internals.base_operator import BaseOperator
from texttools.tools.internals.formatters import Formatter
from texttools.tools.internals.prompt_loader import PromptLoader

# Base Model type for output models
T = TypeVar("T", bound=BaseModel)

# Configure logger
logger = logging.getLogger("operator")
logger.setLevel(logging.INFO)


class Operator(BaseOperator):
    """
    Core engine for running text-processing operations with an LLM (Sync).

    It wires together:
    - `PromptLoader` → loads YAML prompt templates.
    - `UserMergeFormatter` → applies formatting to messages (e.g., merging).
    - OpenAI client → executes completions/parsed completions.
    """

    def __init__(self, client: OpenAI, model: str):
        self.client: OpenAI = client
        self.model = model

    def _analyze(self, prompt_configs: dict[str, str], temperature: float) -> str:
        analyze_prompt = prompt_configs["analyze_template"]
        analyze_message = [self._build_user_message(analyze_prompt)]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=analyze_message,
            temperature=temperature,
        )
        analysis = completion.choices[0].message.content.strip()
        return analysis

    def _parse_completion(
        self,
        message: list[dict[str, str]],
        output_model: Type[T],
        temperature: float,
        logprobs: bool = False,
        top_logprobs: int = 3,
    ) -> tuple[Type[T], Any]:
        request_kwargs = {
            "model": self.model,
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

    def _vllm_completion(
        self,
        message: list[dict[str, str]],
        output_model: Type[T],
        temperature: float,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> tuple[Type[T], Any]:
        json_schema = output_model.model_json_schema()

        # Build kwargs dynamically
        request_kwargs = {
            "model": self.model,
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
        # Internal parameters
        prompt_file: str,
        output_model: Type[T],
        resp_format: Literal["vllm", "parse"],
        mode: str | None,
        **extra_kwargs,
    ) -> dict[str, Any]:
        """
        Execute the LLM pipeline with the given input text.
        """
        prompt_loader = PromptLoader()
        formatter = Formatter()

        try:
            prompt_configs = prompt_loader.load(
                prompt_file=prompt_file,
                text=text.strip(),
                mode=mode,
                **extra_kwargs,
            )

            messages: list[dict[str, str]] = []

            if with_analysis:
                analysis = self._analyze(prompt_configs, temperature)
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
            messages = formatter.user_merge_format(messages)

            if resp_format == "vllm":
                parsed, completion = self._vllm_completion(
                    messages, output_model, temperature, logprobs, top_logprobs
                )
            elif resp_format == "parse":
                parsed, completion = self._parse_completion(
                    messages, output_model, temperature, logprobs, top_logprobs
                )

            # Ensure output_model has a `result` field
            if not hasattr(parsed, "result"):
                logger.error(
                    "The provided output_model must define a field named 'result'"
                )

            result = {"result": parsed.result}

            if logprobs:
                result["logprobs"] = self._extract_logprobs(completion)

            if with_analysis:
                result["analysis"] = analysis

            return result

        except Exception as e:
            logger.error(f"TheTool failed: {e}")
            return {"Error": str(e), "result": ""}
