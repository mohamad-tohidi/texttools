import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel

from ..core import (
    Bool,
    ListDictStrStr,
    ListStr,
    Operator,
    ReasonListStr,
    Str,
    TheToolUtils,
    TokenUsage,
    create_literal_model,
)
from ..models import CategoryTree, ToolOutput, ToolOutputMetadata


class TheTool:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        raise_on_error: bool = True,
    ) -> None:
        """
        Initialize the TheTool instance.

        Args:
            client: An OpenAI client instance for making API calls
            model: The name of the model
            raise_on_error: If True, raises exceptions on errors; if False, logs errors and continues
        """
        self._operator = Operator(client=client, model=model)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.raise_on_error = raise_on_error

    def categorize(
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
    ) -> ToolOutput:
        """
        Classify text into given categories

        Arguments:
            text: The input text
            categories: The category list or category tree
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

        Returns:
            ToolOutput
        """
        tool_name = "categorize"
        start = perf_counter()

        try:
            if isinstance(categories, list):
                operator_output = self._operator.run(
                    # Parameters used for prompt injection
                    text=text,
                    category_list=categories,
                    # Parameters used for chat completions & operator usage
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

                    level_operator_output = self._operator.run(
                        # Parameters used for prompt injection
                        text=text,
                        category_list=category_list,
                        # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def extract_keywords(
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

        Returns:
            ToolOutput
        """
        if mode != "count" and number_of_keywords:
            self.logger.warning(
                "You have set 'number_of_keywords' but didn't use 'count' mode, so it will be ignored"
            )

        tool_name = "extract_keywords"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                number_of_keywords=number_of_keywords,
                mode=mode,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def extract_entities(
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

        Returns:
            ToolOutput
        """
        tool_name = "extract_entities"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                entities=entities,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def is_question(
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

        Returns:
            ToolOutput
        """
        tool_name = "is_question"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def to_question(
        self,
        text: str,
        number_of_questions: int = 1,
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

        Returns:
            ToolOutput
        """
        tool_name = "to_question"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                number_of_questions=number_of_questions,
                mode=mode,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def merge_questions(
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

        Returns:
            ToolOutput
        """
        tool_name = "merge_questions"
        start = perf_counter()

        try:
            text = ", ".join(text)
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                mode=mode,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def augment(
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

        Returns:
            ToolOutput
        """
        tool_name = "augment"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                mode=mode,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def summarize(
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

        Returns:
            ToolOutput
        """
        tool_name = "summarize"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def translate(
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
        priority: int | None = None,
    ) -> ToolOutput:
        """
        Translate text between languages

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            text: The input text
            target_language: The target language for translation
            use_chunker: Whether to use text chunker for large texts
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            user_prompt: Additional instructions
            temperature: Controls randomness
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)

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
                    self.logger.info(f"Processing chunk {i + 1} of the input...")

                    chunk_operator_output = self._operator.run(
                        # Parameters used for prompt injection
                        text=chunk,
                        target_language=target_language,
                        # Parameters used for chat completions & operator usage
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
                operator_output = self._operator.run(
                    # Parameters used for prompt injection
                    text=text,
                    target_language=target_language,
                    # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def propositionize(
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

        Returns:
            ToolOutput
        """
        tool_name = "propositionize"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def is_fact(
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

        Returns:
            ToolOutput
        """
        tool_name = "is_fact"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # Parameters used for prompt injection
                text=text,
                source_text=source_text,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output

    def run_custom(
        self,
        prompt: str,
        output_model: BaseModel,
        with_analysis: bool = False,
        analyze_template: str | None = None,
        output_lang: str | None = None,
        temperature: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
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

        Returns:
            ToolOutput
        """
        tool_name = "run_custom"
        start = perf_counter()

        try:
            operator_output = self._operator.run(
                # User paramaeters
                text=prompt,
                output_model=output_model,
                # Parameters used for chat completions & operator usage
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

        except Exception as e:
            self.logger.error(str(e))

            if self.raise_on_error:
                raise

            metadata = ToolOutputMetadata(tool_name=tool_name)
            tool_output = ToolOutput(
                errors=[f"{type(e).__name__}: {e}"], metadata=metadata
            )

        return tool_output
