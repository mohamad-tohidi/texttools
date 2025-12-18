import sys
from time import perf_counter
from typing import Literal, Any
from collections.abc import Callable

from openai import AsyncOpenAI

from texttools.internals.async_operator import AsyncOperator
import texttools.internals.models as Models
from texttools.internals.exceptions import (
    TextToolsError,
    PromptError,
    LLMError,
    ValidationError,
)
from texttools.internals.text_to_chunks import text_to_chunks


class AsyncTheTool:
    """
    Each method configures the operator with a specific YAML prompt,
    output schema, and flags, then delegates execution to `operator.run()`.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
    ):
        self._operator = AsyncOperator(client=client, model=model)

    async def categorize(
        self,
        text: str,
        categories: list[str] | Models.CategoryTree,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int = 0,
    ) -> Models.ToolOutput:
        """
        Categorize a text into a category / category tree.

        Important Note: category_tree mode is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            text: The input text
            categories: The category list / category tree
            with_analysis: Whether to include detailed reasoning analysis
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput

        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            if isinstance(categories, list):
                operator_output = await self._operator.run(
                    # User parameters
                    text=text,
                    category_list=categories,
                    with_analysis=with_analysis,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    validator=validator,
                    max_validation_retries=max_validation_retries,
                    priority=priority,
                    # Internal parameters
                    tool_name=tool_name,
                    output_model=Models.create_dynamic_model(categories),
                    mode=None,
                    output_lang=None,
                )

                metadata = Models.ToolOutputMetadata(
                    tool_name=tool_name, execution_time=perf_counter() - start
                )
                tool_output = Models.ToolOutput(
                    result=operator_output.result,
                    analysis=operator_output.analysis,
                    logprobs=operator_output.logprobs,
                    metadata=metadata,
                )

            else:
                levels = categories.get_level_count()
                parent_node = categories.get_node("root")
                final_categories = []
                analysis = ""
                logprobs_list = []

                for _ in range(levels):
                    if not parent_node.children:
                        break

                    category_list = [
                        f"Category Name: {name}, Description: {node.description}"
                        for name, node in parent_node.children.items()
                    ]
                    category_names = list(parent_node.children.keys())

                    level_operator_output = await self._operator.run(
                        # User parameters
                        text=text,
                        category_list=category_list,
                        with_analysis=with_analysis,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        validator=validator,
                        max_validation_retries=max_validation_retries,
                        priority=priority,
                        # Internal parameters
                        tool_name=tool_name,
                        output_model=Models.create_dynamic_model(category_names),
                        mode=None,
                        output_lang=None,
                    )

                    chosen_category = level_operator_output.result
                    parent_node = categories.get_node(chosen_category)
                    if not parent_node:
                        break
                    final_categories.append(chosen_category)

                    if with_analysis:
                        analysis += level_operator_output.analysis
                    if logprobs:
                        logprobs_list.extend(level_operator_output.logprobs)

                metadata = Models.ToolOutputMetadata(
                    tool_name=tool_name, execution_time=(perf_counter() - start)
                )
                tool_output = Models.ToolOutput(
                    result=final_categories,
                    analysis=analysis,
                    logprobs=logprobs_list,
                    metadata=metadata,
                )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def extract_keywords(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        mode: Literal["auto", "threshold", "count"] = "auto",
        number_of_keywords: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Extract salient keywords from text.

        Arguments:
            text: The input text
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                mode=mode,
                number_of_keywords=number_of_keywords,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.ListStr,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def extract_entities(
        self,
        text: str,
        entities: list[str] | None = None,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Perform Named Entity Recognition (NER) over the input text.

        Arguments:
            text: The input text
            entities: List of entities provided by user (Optional)
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                entities=entities
                or "all named entities (e.g., PER, ORG, LOC, DAT, etc.)",
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.ListDictStrStr,
                mode=None,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def is_question(
        self,
        text: str,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Detect if the input is phrased as a question.

        Arguments:
            text: The input text
            with_analysis: Whether to include detailed reasoning analysis
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name

        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                with_analysis=with_analysis,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.Bool,
                mode=None,
                output_lang=None,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def text_to_question(
        self,
        text: str,
        number_of_questions: int,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Generate a single question from the given text.

        Arguments:
            text: The input text
            number_of_questions: Number of questions to generate
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                number_of_questions=number_of_questions,
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.ReasonListStr,
                mode=None,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def merge_questions(
        self,
        text: list[str],
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        mode: Literal["default", "reason"] = "default",
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Merge multiple questions into a single unified question.

        Arguments:
            text: List of questions to merge
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            text = ", ".join(text)
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.Str,
                mode=mode,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def rewrite(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        mode: Literal["positive", "negative", "hard_negative"] = "positive",
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Rewrite a text with different modes.

        Arguments:
            text: The input text
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.Str,
                mode=mode,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def subject_to_question(
        self,
        text: str,
        number_of_questions: int,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Generate a list of questions about a subject.

        Arguments:
            text: The subject text to generate questions about
            number_of_questions: Number of questions to generate
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                number_of_questions=number_of_questions,
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.ReasonListStr,
                mode=None,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def summarize(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Summarize the given subject text.

        Arguments:
            text: The input text
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.Str,
                mode=None,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def translate(
        self,
        text: str,
        target_language: str,
        use_chunker: bool = True,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Translate text between languages.

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            text: The input text
            target_language: The target language for translation
            use_chunker: Whether to use text chunker for text length bigger than 1500
            with_analysis: Whether to include detailed reasoning analysis
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            if len(text.split(" ")) > 1500 and use_chunker:
                chunks = text_to_chunks(text, 1200, 0)
                translation = ""
                analysis = ""
                logprobs_list = []

                for chunk in chunks:
                    chunk_operator_output = await self._operator.run(
                        # User parameters
                        text=chunk,
                        target_language=target_language,
                        with_analysis=with_analysis,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        validator=validator,
                        max_validation_retries=max_validation_retries,
                        priority=priority,
                        # Internal parameters
                        tool_name=tool_name,
                        output_model=Models.Str,
                        mode=None,
                        output_lang=None,
                    )

                    translation += chunk_operator_output.result + "\n"

                    if with_analysis:
                        analysis += chunk_operator_output.analysis
                    if logprobs:
                        logprobs_list.extend(chunk_operator_output.logprobs)

                metadata = Models.ToolOutputMetadata(
                    tool_name=tool_name, execution_time=perf_counter() - start
                )
                tool_output = Models.ToolOutput(
                    result=translation,
                    logprobs=logprobs_list,
                    analysis=analysis,
                    metadata=metadata,
                )

            else:
                operator_output = await self._operator.run(
                    # User parameters
                    text=text,
                    target_language=target_language,
                    with_analysis=with_analysis,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    validator=validator,
                    max_validation_retries=max_validation_retries,
                    priority=priority,
                    # Internal parameters
                    tool_name=tool_name,
                    output_model=Models.Str,
                    mode=None,
                    output_lang=None,
                )

                metadata = Models.ToolOutputMetadata(
                    tool_name=tool_name, execution_time=perf_counter() - start
                )
                tool_output = Models.ToolOutput(
                    result=operator_output.result,
                    logprobs=operator_output.logprobs,
                    analysis=operator_output.analysis,
                    metadata=metadata,
                )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def propositionize(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Proposition input text to meaningful sentences.

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            text: The input text
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.ListStr,
                mode=None,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def check_fact(
        self,
        text: str,
        source_text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Checks wheather a statement is relevant to the source text or not.

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            text: The input text
            source_text: the source text that we want to check relation of text to it
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output
            user_prompt: Additional instructions
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User parameters
                text=text,
                with_analysis=with_analysis,
                output_lang=output_lang,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                output_model=Models.Bool,
                mode=None,
                source_text=source_text,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def run_custom(
        self,
        prompt: str,
        output_model: Any,
        with_analysis: bool = False,
        analyze_template: str | None = None,
        output_lang: str | None = None,
        temperature: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Custom tool that can do almost anything!

        Arguments:
            prompt: The user prompt
            output_model: Pydantic BaseModel used for structured output
            with_analysis: Whether to include detailed reasoning analysis
            analyze_template: The analyze template used for reasoning analysis
            output_lang: Language for the output
            temperature: Controls randomness (0.0 - 2.0)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = sys._getframe().f_code.co_name
        start = perf_counter()

        try:
            operator_output = await self._operator.run(
                # User paramaeters
                text=prompt,
                output_model=output_model,
                with_analysis=with_analysis,
                analyze_template=analyze_template,
                output_model_str=output_model.model_json_schema(),
                output_lang=output_lang,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                validator=validator,
                max_validation_retries=max_validation_retries,
                priority=priority,
                # Internal parameters
                tool_name=tool_name,
                user_prompt=None,
                mode=None,
            )

            metadata = Models.ToolOutputMetadata(
                tool_name=tool_name, execution_time=perf_counter() - start
            )
            tool_output = Models.ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            metadata = Models.ToolOutputMetadata(tool_name=tool_name)
            tool_output = Models.ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output
