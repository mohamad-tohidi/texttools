import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

from openai import AsyncOpenAI

from ..core import (
    AsyncOperator,
    Bool,
    ListDictStrStr,
    ListStr,
    LLMError,
    PromptError,
    ReasonListStr,
    Str,
    TextToolsError,
    TheToolUtils,
    TokenUsage,
    ValidationError,
    create_literal_model,
)
from ..models import CategoryTree, ToolOutput, ToolOutputMetadata


class AsyncTheTool:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        raise_on_error: bool = True,
    ):
        """
        Initialize the AsyncTheTool instance.

        Args:
            client: An AsyncOpenAI client instance for making asynchronous API calls
            model: The name of the model
            raise_on_error: If True, raises exceptions on errors; if False, logs errors and continues
        """
        self._operator = AsyncOperator(client=client, model=model)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.raise_on_error = raise_on_error

    async def categorize(
        self,
        text: str,
        categories: list[str] | CategoryTree,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Classify text into given categories

        Arguments:
            text: The input text
            categories: The category list / category tree
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "categorize"
        start = perf_counter()

        try:
            if isinstance(categories, list):
                operator_output = await TheToolUtils.run_with_timeout(
                    self._operator.run(
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
                        output_model=create_literal_model(categories),
                        mode=None,
                        output_lang=None,
                    ),
                    timeout=timeout,
                )

                metadata = ToolOutputMetadata(
                    tool_name=tool_name,
                    execution_time=perf_counter() - start,
                    processed_by=operator_output.processed_by,
                    token_usage=operator_output.token_usage,
                )
                tool_output = ToolOutput(
                    result=operator_output.result,
                    analysis=operator_output.analysis,
                    logprobs=operator_output.logprobs,
                    metadata=metadata,
                )

            else:
                max_depth = categories.get_max_depth()
                parent_node = categories.get_node("root")
                final_categories = []
                analysis = ""
                logprobs_list = []
                token_usage = TokenUsage()

                for level in range(max_depth):
                    self.logger.info(f"Processing level {level + 1} of the tree...")

                    if not parent_node.children:
                        break

                    category_list = [
                        f"Category Name: {name}, Description: {node.description}"
                        for name, node in parent_node.children.items()
                    ]
                    category_names = list(parent_node.children.keys())

                    self.logger.info(
                        f"Categories available in the current level: {category_names}"
                    )

                    level_operator_output = await TheToolUtils.run_with_timeout(
                        self._operator.run(
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
                            output_model=create_literal_model(category_names),
                            mode=None,
                            output_lang=None,
                        ),
                        timeout=timeout,
                    )

                    chosen_category = level_operator_output.result
                    parent_node = categories.get_node(chosen_category)

                    self.logger.info(
                        f"Chosen category in the current level: {chosen_category}"
                    )

                    if not parent_node:
                        break

                    final_categories.append(chosen_category)

                    if with_analysis:
                        analysis += level_operator_output.analysis
                    if logprobs:
                        logprobs_list.extend(level_operator_output.logprobs)
                    token_usage += level_operator_output.token_usage

                metadata = ToolOutputMetadata(
                    tool_name=tool_name,
                    execution_time=perf_counter() - start,
                    processed_by=level_operator_output.processed_by,
                    token_usage=token_usage,
                )
                tool_output = ToolOutput(
                    result=final_categories,
                    analysis=analysis,
                    logprobs=logprobs_list,
                    metadata=metadata,
                )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def extract_keywords(
        self,
        text: str,
        mode: Literal["auto", "threshold", "count"] = "auto",
        number_of_keywords: int | None = None,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Extract keywords from the text

        Arguments:
            text: The input text
            mode: auto -> decide n of keywords automatically, threshold -> decide n of keywords by a threshold, count -> takes number of keywords as the parameter
            number_of_keywords: Must be set only when using "count" mode
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            output_lang: Forces the model to respond in a specific language
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "extract_keywords"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
                    # User parameters
                    text=text,
                    with_analysis=with_analysis,
                    number_of_keywords=number_of_keywords,
                    mode=mode,
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
                    output_model=ListStr,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def extract_entities(
        self,
        text: str,
        entities: list[str] = ["all named entities"],
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Perform Named Entity Recognition (NER)

        Arguments:
            text: The input text
            entities: List of entities
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            output_lang: Forces the model to respond in a specific language
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "extract_entities"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
                    # User parameters
                    text=text,
                    entities=entities,
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
                    output_model=ListDictStrStr,
                    mode=None,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
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
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Detect if the input is phrased as a question.

        Arguments:
            text: The input text
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "is_question"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
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
                    output_model=Bool,
                    mode=None,
                    output_lang=None,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def to_question(
        self,
        text: str,
        number_of_questions: int,
        mode: Literal["from_text", "from_subject"] = "from_text",
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Generate questions from the given text / subject

        Arguments:
            text: The input text
            mode: from_text -> generate questions from an answer, from_subject -> generate questions from a subject
            number_of_questions: Number of questions to generate
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            output_lang: Forces the model to respond in a specific language
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "to_question"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
                    # User parameters
                    text=text,
                    number_of_questions=number_of_questions,
                    mode=mode,
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
                    output_model=ReasonListStr,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def merge_questions(
        self,
        text: list[str],
        mode: Literal["simple", "stepwise"] = "simple",
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Merge multiple questions into a single unified question

        Arguments:
            text: List of questions to merge
            mode: simple -> regular question merging, stepwise -> merge questions in two steps
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            output_lang: Forces the model to respond in a specific language
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "merge_questions"
        start = perf_counter()

        try:
            text = ", ".join(text)
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
                    # User parameters
                    text=text,
                    mode=mode,
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
                    output_model=Str,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def augment(
        self,
        text: str,
        mode: Literal["positive", "negative", "hard_negative"] = "positive",
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Rewrite text in different augmentations

        Arguments:
            text: The input text
            mode: positive -> positive augmentation, negative -> negative augmentation, hard_negative -> hard negative augmentation
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            output_lang: Forces the model to respond in a specific language
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "augment"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
                    # User parameters
                    text=text,
                    mode=mode,
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
                    output_model=Str,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
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
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Summarize the given text

        Arguments:
            text: The input text
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            output_lang: Forces the model to respond in a specific language
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "summarize"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
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
                    output_model=Str,
                    mode=None,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def translate(
        self,
        text: str,
        target_lang: str,
        use_chunker: bool = True,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Translate text between languages

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            text: The input text
            target_lang: The target language for translation
            use_chunker: Whether to use text chunker for large texts
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "translate"
        start = perf_counter()

        try:
            if len(text.split(" ")) > 1500 and use_chunker:
                self.logger.info("Running translator using chunker...")

                chunks = TheToolUtils.to_chunks(text, 1200, 0)
                translation = ""
                analysis = ""
                logprobs_list = []
                token_usage = TokenUsage()

                for i, chunk in enumerate(chunks):
                    self.logger.info(f"Processing chunk {i} of the input...")

                    chunk_operator_output = await TheToolUtils.run_with_timeout(
                        self._operator.run(
                            # User parameters
                            text=chunk,
                            target_lang=target_lang,
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
                            output_model=Str,
                            mode=None,
                            output_lang=None,
                        ),
                        timeout=timeout,
                    )

                    translation += chunk_operator_output.result + "\n"

                    if with_analysis:
                        analysis += chunk_operator_output.analysis
                    if logprobs:
                        logprobs_list.extend(chunk_operator_output.logprobs)
                    token_usage += chunk_operator_output.token_usage

                metadata = ToolOutputMetadata(
                    tool_name=tool_name,
                    execution_time=perf_counter() - start,
                    processed_by=chunk_operator_output.processed_by,
                    token_usage=token_usage,
                )
                tool_output = ToolOutput(
                    result=translation,
                    logprobs=logprobs_list,
                    analysis=analysis,
                    metadata=metadata,
                )

            else:
                operator_output = await TheToolUtils.run_with_timeout(
                    self._operator.run(
                        # User parameters
                        text=text,
                        target_lang=target_lang,
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
                        output_model=Str,
                        mode=None,
                        output_lang=None,
                    ),
                    timeout=timeout,
                )

                metadata = ToolOutputMetadata(
                    tool_name=tool_name,
                    execution_time=perf_counter() - start,
                    processed_by=operator_output.processed_by,
                    token_usage=operator_output.token_usage,
                )
                tool_output = ToolOutput(
                    result=operator_output.result,
                    logprobs=operator_output.logprobs,
                    analysis=operator_output.analysis,
                    metadata=metadata,
                )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
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
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Convert a text into atomic, independent, meaningful sentences

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            text: The input text
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            output_lang: Forces the model to respond in a specific language
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "propositionize"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
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
                    output_model=ListStr,
                    mode=None,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    async def is_fact(
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
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Check whether a statement is a fact based on the source text

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            text: The input text
            source_text: The source text
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            output_lang: Forces the model to respond in a specific language
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "is_fact"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
                    # User parameters
                    text=text,
                    source_text=source_text,
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
                    output_model=Bool,
                    mode=None,
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
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
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ToolOutput:
        """
        Custom tool that can do almost anything

        Arguments:
            prompt: The user prompt
            output_model: Pydantic BaseModel used for structured output
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            analyze_template: The analyze template used for reasoning analysis
            output_lang: Forces the model to respond in a specific language
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            ToolOutput
        """
        tool_name = "run_custom"
        start = perf_counter()

        try:
            operator_output = await TheToolUtils.run_with_timeout(
                self._operator.run(
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
                ),
                timeout=timeout,
            )

            metadata = ToolOutputMetadata(
                tool_name=tool_name,
                execution_time=perf_counter() - start,
                processed_by=operator_output.processed_by,
                token_usage=operator_output.token_usage,
            )
            tool_output = ToolOutput(
                result=operator_output.result,
                logprobs=operator_output.logprobs,
                analysis=operator_output.analysis,
                metadata=metadata,
            )

        except (PromptError, LLMError, ValidationError, TextToolsError, Exception) as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output
