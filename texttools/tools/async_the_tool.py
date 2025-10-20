from typing import Literal, Any

from openai import AsyncOpenAI

from texttools.tools.internals.async_operator import AsyncOperator
import texttools.tools.internals.output_models as OutputModels


class AsyncTheTool:
    """
    Async counterpart to TheTool.

    Usage:
        async_client = AsyncOpenAI(...)
        tool = TheToolAsync(async_client, model="model-name")
        result = await tool.categorize("text ...", with_analysis=True)
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
    ):
        self.operator = AsyncOperator(client=client, model=model)

    async def categorize(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, str]:
        """
        Categorize a text into a single Islamic studies domain category.

        Args:
            text: Input string to categorize.
            with_analysis: If True, first runs an LLM "analysis" step and
                           conditions the main prompt on that analysis.

        Returns:
            {"result": <category string>}
            Example: {"result": "باورهای دینی"}
        """
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="categorizer.yaml",
            output_model=OutputModels.CategorizerOutput,
            resp_format="parse",
            mode=None,
        )

    async def extract_keywords(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, list[str]]:
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="extract_keywords.yaml",
            output_model=OutputModels.ListStrOutput,
            resp_format="parse",
            mode=None,
        )

    async def extract_entities(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="extract_entities.yaml",
            output_model=OutputModels.ListDictStrStrOutput,
            resp_format="parse",
            mode=None,
        )

    async def is_question(
        self,
        text: str,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, bool]:
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="is_question.yaml",
            output_model=OutputModels.BoolOutput,
            resp_format="parse",
            mode=None,
            output_lang=None,
        )

    async def text_to_question(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, str]:
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="text_to_question.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            mode=None,
        )

    async def merge_questions(
        self,
        text: list[str],
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        mode: Literal["default", "reason"] = "default",
    ) -> dict[str, str]:
        text = ", ".join(text)
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="merge_questions.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            mode=mode,
        )

    async def rewrite(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        mode: Literal["positive", "negative", "hard_negative"] = "positive",
    ) -> dict[str, str]:
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="rewrite.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            mode=mode,
        )

    async def subject_to_question(
        self,
        text: str,
        number_of_questions: int,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, list[str]]:
        return await self.operator.run(
            # User parameters
            text=text,
            number_of_questions=number_of_questions,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="subject_to_question.yaml",
            output_model=OutputModels.ReasonListStrOutput,
            resp_format="parse",
            mode=None,
        )

    async def summarize(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, str]:
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="summarize.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            mode=None,
        )

    async def translate(
        self,
        text: str,
        target_language: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, str]:
        return await self.operator.run(
            # User parameters
            text=text,
            target_language=target_language,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="translate.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            mode=None,
        )

    async def run_custom(
        self,
        prompt: str,
        output_model: Any,
        output_lang: str | None = None,
        temperature: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> dict[str, Any]:
        """
        Custom tool that can do almost anything!

        Args:
            prompt: Custom prompt.
            output_model: Custom BaseModel output model.

        Returns:
            {"result": <Any>}
        """
        return await self.operator.run(
            # User paramaeters
            text=prompt,
            output_model=output_model,
            output_model_str=output_model.model_json_schema(),
            output_lang=output_lang,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="run_custom.yaml",
            resp_format="parse",
            user_prompt=None,
            with_analysis=False,
            mode=None,
        )
