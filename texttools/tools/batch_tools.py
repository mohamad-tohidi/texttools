import asyncio
import logging
from collections.abc import Callable
from typing import Any, Literal

from openai import AsyncOpenAI

from ..models import CategoryTree, ToolOutput
from .async_tools import AsyncTheTool


class BatchTheTool:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        raise_on_error: bool = True,
        max_concurrency: int = 5,
    ) -> None:
        """
        Initialize the BatchTheTool instance.

        Arguments:
            client: An AsyncOpenAI client instance for making asynchronous API calls
            model: The name of the model
            raise_on_error: If True, raises exceptions on errors; if False, logs errors and continues
            max_concurrency: Maximum number of concurrent API requests allowed
        """
        self.tool = AsyncTheTool(client, model, raise_on_error)
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def categorize(
        self,
        texts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Classify texts into given categories

        Arguments:
            texts: The input texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.categorize(
                    text=text,
                    categories=categories,
                    with_analysis=with_analysis,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    validator=validator,
                    max_validation_retries=max_validation_retries,
                    priority=priority,
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def extract_keywords(
        self,
        texts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Extract keywords from the texts

        Arguments:
            texts: The input texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.extract_keywords(
                    text=text,
                    mode=mode,
                    number_of_keywords=number_of_keywords,
                    with_analysis=with_analysis,
                    output_lang=output_lang,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    validator=validator,
                    max_validation_retries=max_validation_retries,
                    priority=priority,
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def extract_entities(
        self,
        texts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Perform Named Entity Recognition (NER) on texts

        Arguments:
            texts: The input texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.extract_entities(
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
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def is_question(
        self,
        texts: list[str],
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> list[ToolOutput]:
        """
        Detect if the inputs are phrased as questions.

        Arguments:
            texts: The input texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.is_question(
                    text=text,
                    with_analysis=with_analysis,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    validator=validator,
                    max_validation_retries=max_validation_retries,
                    priority=priority,
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def to_question(
        self,
        texts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Generate questions from the given texts / subjects

        Arguments:
            texts: The input texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.to_question(
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
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def merge_questions(
        self,
        texts: list[list[str]],
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
    ) -> list[ToolOutput]:
        """
        Merge multiple questions into a single unified question for each group

        Arguments:
            texts: List of groups of questions to merge
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(texts: list[str]) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.merge_questions(
                    text=texts,
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
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def augment(
        self,
        texts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Rewrite texts in different augmentations

        Arguments:
            texts: The input texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.augment(
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
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def summarize(
        self,
        texts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Summarize the given texts

        Arguments:
            texts: The input texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.summarize(
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
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def translate(
        self,
        texts: list[str],
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
        timeout: float | None = None,
    ) -> list[ToolOutput]:
        """
        Translate texts between languages

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            texts: The input texts
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
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.translate(
                    text=text,
                    target_language=target_language,
                    use_chunker=use_chunker,
                    with_analysis=with_analysis,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    validator=validator,
                    max_validation_retries=max_validation_retries,
                    priority=priority,
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def propositionize(
        self,
        texts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Convert texts into atomic, independent, meaningful sentences

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            texts: The input texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.propositionize(
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
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t) for t in texts]
        return await asyncio.gather(*tasks)

    async def is_fact(
        self,
        texts: list[str],
        source_texts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Check whether statements are facts based on source texts

        Important Note: This tool is EXPERIMENTAL, you can use it but it isn't reliable.

        Arguments:
            texts: The input texts (statements to check)
            source_texts: The source texts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(texts)} texts...")

        processed = 0
        total = len(texts)

        async def _throttled_task(text: str, source_text: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.is_fact(
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
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(t, s) for t, s in zip(texts, source_texts)]
        return await asyncio.gather(*tasks)

    async def run_custom(
        self,
        prompts: list[str],
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
    ) -> list[ToolOutput]:
        """
        Custom tool that can do almost anything for multiple prompts

        Arguments:
            prompts: The user prompts
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
            list[ToolOutput]
        """

        self.logger.info(f"Starting batch tool with {len(prompts)} prompts...")

        processed = 0
        total = len(prompts)

        async def _throttled_task(prompt: str) -> ToolOutput:
            nonlocal processed

            async with self.semaphore:
                result = await self.tool.run_custom(
                    prompt=prompt,
                    output_model=output_model,
                    with_analysis=with_analysis,
                    analyze_template=analyze_template,
                    output_lang=output_lang,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    validator=validator,
                    max_validation_retries=max_validation_retries,
                    priority=priority,
                    timeout=timeout,
                )
            processed += 1
            if processed % self.max_concurrency == 0 or processed == total:
                self.logger.info(f"Processed {processed}/{total}")
            return result

        tasks = [_throttled_task(p) for p in prompts]
        return await asyncio.gather(*tasks)
