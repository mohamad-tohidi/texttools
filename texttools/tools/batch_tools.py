import asyncio
import logging
from collections.abc import Callable
from typing import Any, Awaitable, Literal

from openai import AsyncOpenAI
from tqdm import tqdm

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

    async def _run_batch(
        self,
        inputs: list[str | tuple | list],
        coro: Callable[..., Awaitable[ToolOutput]],
        desc: str,
        **fixed_kwargs,
    ) -> list[ToolOutput]:
        """
        Process a batch of inputs with throttled concurrency and a progress bar.

        Args:
            inputs: List where each element is either a single argument or multiple arguments.
            coro: The async tool method to call (e.g., self.tool.categorize).
            desc: Description shown in the progress bar.

        Returns:
            List of ToolOutput objects.
        """
        total = len(inputs)
        with tqdm(total=total, desc=desc, unit="text") as pbar:

            async def throttled_task(*args):
                async with self.semaphore:
                    result = await coro(*args, **fixed_kwargs)
                    pbar.update(1)
                    return result

            tasks = []
            for inp in inputs:
                if isinstance(inp, (tuple, list)):
                    tasks.append(throttled_task(*inp))
                else:
                    tasks.append(throttled_task(inp))

            results = await asyncio.gather(*tasks)
        return results

    async def categorize(
        self,
        texts: list[str],
        categories: list[str] | CategoryTree,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> list[ToolOutput]:
        """
        Classify texts into given categories

        Arguments:
            texts: The input texts
            categories: The category list or category tree
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            user_prompt: Additional instructions
            temperature: Controls randomness
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: str | list[str] - The predicted category label (or list of labels when using a category tree) for each input text
        """
        self.logger.info(f"Starting batch categorize with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.categorize,
            desc="Categorizing...",
            categories=categories,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def extract_keywords(
        self,
        texts: list[str],
        mode: Literal["auto", "threshold", "count"] = "auto",
        number_of_keywords: int | None = None,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: list[str] - List of extracted keywords for each input text
        """
        self.logger.info(f"Starting batch extract_keywords with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.extract_keywords,
            desc="Extracting Keywords...",
            mode=mode,
            number_of_keywords=number_of_keywords,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def extract_entities(
        self,
        texts: list[str],
        entities: list[str] = ["all named entities"],
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: list[dict[str, str]] - List of dictionaries containing entity types and their text spans for each input text
        """
        self.logger.info(f"Starting batch extract_entities with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.extract_entities,
            desc="Extracting Entities...",
            entities=entities,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def is_question(
        self,
        texts: list[str],
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: bool - True if the input is a question, False otherwise, for each input text
        """
        self.logger.info(f"Starting batch is_question with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.is_question,
            desc="Detecting Questions...",
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def to_question(
        self,
        texts: list[str],
        number_of_questions: int = 1,
        mode: Literal["from_text", "from_subject"] = "from_text",
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: list[str] - List of generated questions for each input text
        """
        self.logger.info(f"Starting batch to_question with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.to_question,
            desc="Generating Questions...",
            number_of_questions=number_of_questions,
            mode=mode,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def merge_questions(
        self,
        texts: list[list[str]],
        mode: Literal["simple", "stepwise"] = "simple",
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: str - The merged question for each group
        """
        self.logger.info(f"Starting batch merge_questions with {len(texts)} groups...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.merge_questions,
            desc="Merging Questions...",
            mode=mode,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def augment(
        self,
        texts: list[str],
        mode: Literal["positive", "negative", "hard_negative"] = "positive",
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: str - The augmented (rewritten) text for each input text
        """
        self.logger.info(f"Starting batch augment with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.augment,
            desc="Augmenting...",
            mode=mode,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def summarize(
        self,
        texts: list[str],
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: str - The generated summary for each input text
        """
        self.logger.info(f"Starting batch summarize with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.summarize,
            desc="Summarizing...",
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def translate(
        self,
        texts: list[str],
        target_language: str,
        use_chunker: bool = True,
        max_concurrent_chunks: int = 5,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
        priority: int | None = None,
        timeout: float | None = None,
    ) -> list[ToolOutput]:
        """
        Translate texts between languages

        Arguments:
            texts: The input texts
            target_language: The target language for translation
            use_chunker: Whether to use text chunker for large texts
            max_concurrent_chunks: Maximum number of chunks to process in parallel when chunking is enabled
            with_analysis: Adds a reasoning step before generating the final output. Note: This doubles token usage per call
            user_prompt: Additional instructions
            temperature: Controls randomness
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: str - The translated text for each input text
        """
        self.logger.info(f"Starting batch translate with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.translate,
            desc="Translating...",
            target_language=target_language,
            use_chunker=use_chunker,
            max_concurrent_chunks=max_concurrent_chunks,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def propositionize(
        self,
        texts: list[str],
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: list[str] - List of atomic propositions (independent meaningful sentences) for each input text
        """
        self.logger.info(f"Starting batch propositionize with {len(texts)} texts...")

        return await self._run_batch(
            inputs=texts,
            coro=self.tool.propositionize,
            desc="Propositionizing...",
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )

    async def is_fact(
        self,
        texts: list[str],
        source_texts: list[str],
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.0,
        normalize: bool = True,
        logprobs: bool = False,
        top_logprobs: int = 3,
        max_completion_tokens: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int = 3,
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
            normalize: Whether to apply text normalization before sending to the LLM
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            max_completion_tokens: Maximum number of tokens to generate in the completion
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and the model)
            timeout: Maximum time in seconds to wait for the response before raising a timeout error

        Returns:
            list[ToolOutput]
                result: bool - True if the statement is supported by the source text, False otherwise, for each pair
        """
        self.logger.info(f"Starting batch is_fact with {len(texts)} texts...")

        inputs = list(zip(texts, source_texts))
        return await self._run_batch(
            inputs=inputs,
            coro=self.tool.is_fact,
            desc="Checking Facts...",
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            normalize=normalize,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            timeout=timeout,
        )
