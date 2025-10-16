from typing import Literal

from openai import AsyncOpenAI

import texttools.tools.internals.output_models as OutputModels
from texttools.tools.internals.async_operator import AsyncOperator


class AsyncTheTool:
    """
    Async counterpart to TheTool.

    Usage:
        async_client = AsyncOpenAI(...)
        tool = TheToolAsync(async_client, model="gemma-3")
        result = await tool.categorize("متن ...", with_analysis=True)
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        temperature: float = 0.0,
    ):
        self.operator = AsyncOperator(
            client=client,
            model=model,
            temperature=temperature,
        )

    async def categorize(
        self,
        text: str,
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 8,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        results = await self.operator.run(
            text,
            prompt_file="categorizer.yaml",
            output_model=OutputModels.CategorizerOutput,
            with_analysis=with_analysis,
            resp_format="parse",
            user_prompt=user_prompt,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def extract_keywords(
        self,
        text: str,
        output_lang: str | None = None,
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> dict[str, list[str]]:
        results = await self.operator.run(
            text,
            prompt_file="keyword_extractor.yaml",
            output_model=OutputModels.ListStrOutput,
            with_analysis=with_analysis,
            resp_format="parse",
            user_prompt=user_prompt,
            output_lang=output_lang,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def extract_entities(
        self,
        text: str,
        output_lang: str | None = None,
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        results = await self.operator.run(
            text,
            prompt_file="ner_extractor.yaml",
            output_model=OutputModels.ListDictStrStrOutput,
            with_analysis=with_analysis,
            resp_format="parse",
            user_prompt=user_prompt,
            output_lang=output_lang,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def is_question(
        self,
        question: str,
        output_lang: str | None = None,
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 2,
        max_tokens: int | None = None,
    ) -> dict[str, bool]:
        results = await self.operator.run(
            question,
            prompt_file="is_question.yaml",
            output_model=OutputModels.BoolOutput,
            with_analysis=with_analysis,
            resp_format="parse",
            user_prompt=user_prompt,
            output_lang=output_lang,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def text_to_question(
        self,
        text: str,
        output_lang: str | None = None,
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        results = await self.operator.run(
            text,
            prompt_file="text_to_question.yaml",
            output_model=OutputModels.StrOutput,
            with_analysis=with_analysis,
            resp_format="parse",
            user_prompt=user_prompt,
            output_lang=output_lang,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def merge_questions(
        self,
        questions: list[str],
        output_lang: str | None = None,
        mode: Literal["default", "reason"] = "default",
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        question_str = ", ".join(questions)
        results = await self.operator.run(
            question_str,
            prompt_file="question_merger.yaml",
            output_model=OutputModels.StrOutput,
            with_analysis=with_analysis,
            use_modes=True,
            mode=mode,
            resp_format="parse",
            user_prompt=user_prompt,
            output_lang=output_lang,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def rewrite(
        self,
        question: str,
        output_lang: str | None = None,
        mode: Literal["positive", "negative", "hard_negative"] = "positive",
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        results = await self.operator.run(
            question,
            prompt_file="rewriter.yaml",
            output_model=OutputModels.StrOutput,
            with_analysis=with_analysis,
            use_modes=True,
            mode=mode,
            resp_format="parse",
            user_prompt=user_prompt,
            output_lang=output_lang,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def subject_to_question(
        self,
        subject: str,
        number_of_questions: int,
        output_lang: str | None = None,
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> dict[str, list[str]]:
        results = await self.operator.run(
            subject,
            prompt_file="subject_to_question.yaml",
            output_model=OutputModels.ReasonListStrOutput,
            with_analysis=with_analysis,
            resp_format="parse",
            user_prompt=user_prompt,
            number_of_questions=number_of_questions,
            output_lang=output_lang,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def summarize(
        self,
        text: str,
        output_lang: str | None = None,
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        results = await self.operator.run(
            text,
            prompt_file="summarizer.yaml",
            output_model=OutputModels.StrOutput,
            with_analysis=with_analysis,
            resp_format="parse",
            user_prompt=user_prompt,
            output_lang=output_lang,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results

    async def translate(
        self,
        text: str,
        target_language: str,
        with_analysis: bool = False,
        user_prompt: str = "",
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        results = await self.operator.run(
            text,
            prompt_file="translator.yaml",
            output_model=OutputModels.StrOutput,
            with_analysis=with_analysis,
            resp_format="parse",
            user_prompt=user_prompt,
            target_language=target_language,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )
        return results
