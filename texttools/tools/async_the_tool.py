from typing import Literal, Any

from openai import AsyncOpenAI

from texttools.tools.internals.async_operator import AsyncOperator
import texttools.tools.internals.output_models as OutputModels


class AsyncTheTool:
    """
    Async counterpart to TheTool.

    Each method configures the async operator with a specific YAML prompt,
    output schema, and flags, then delegates execution to `operator.run()`.

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
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, str]:
        """
        Categorize a text into a single Islamic studies domain category.

        Returns:
            {"result": <category string>} + ("logprobs" and "analysis" if enabled)
        """
        return await self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="categorizer.yaml",
            output_model=OutputModels.CategorizerOutput,
            resp_format="parse",
            mode=None,
            output_lang=None,
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
        """
        Extract salient keywords from text.

        Returns:
            {"result": [<keyword1>, <keyword2>, ...]} + ("logprobs" and "analysis" if enabled)
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
        """
        Perform Named Entity Recognition (NER) over the input text.

        Returns:
            {"result": [{"text": <entity>, "type": <entity_type>}, ...]} + ("logprobs" and "analysis" if enabled)
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
        """
        Detect if the input is phrased as a question.

        Returns:
            {"result": True} or {"result": False} + ("logprobs" and "analysis" if enabled)
        """
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
        """
        Generate a single question from the given text.

        Returns:
            {"result": <generated_question>} + ("logprobs" and "analysis" if enabled)
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
        """
        Merge multiple questions into a single unified question.

        Returns:
            {"result": <merged_question>} + ("logprobs" and "analysis" if enabled)
        """
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
        """
        Rewrite a text with different modes.

        Returns:
            {"result": <rewritten_text>} + ("logprobs" and "analysis" if enabled)
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
        """
        Generate a list of questions about a subject.

        Returns:
            {"result": [<question1>, <question2>, ...]} + ("logprobs" and "analysis" if enabled)
        """
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
        """
        Summarize the given subject text.

        Returns:
            {"result": <summary>} + ("logprobs" and "analysis" if enabled)
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
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> dict[str, str]:
        """
        Translate text between languages.

        Returns:
            {"result": <translated_text>} + ("logprobs" and "analysis" if enabled)
        """
        return await self.operator.run(
            # User parameters
            text=text,
            target_language=target_language,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="translate.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            mode=None,
            output_lang=None,
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
