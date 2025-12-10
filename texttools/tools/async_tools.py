from datetime import datetime
from typing import Literal, Any
from collections.abc import Callable

from openai import AsyncOpenAI

from texttools.tools.internals.async_operator import AsyncOperator
import texttools.tools.internals.models as Models


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
        self._operator = AsyncOperator(client=client, model=model)

    async def categorize(
        self,
        text: str,
        categories: list[str] | Models.CategoryTree,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        mode: Literal["category_list", "category_tree"] = "category_list",
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Categorize a text into a category / category tree.

        Arguments:
            text: The input text to categorize
            categories: The category / category_tree to give to LLM
            with_analysis: Whether to include detailed reasoning analysis
            user_prompt: Additional instructions for the categorization
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (str): The assigned category
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()

        if mode == "category_tree":
            # Initializations
            output = Models.ToolOutput()
            levels = categories.get_level_count()
            parent_id = 0
            final_output = []

            for _ in range(levels):
                # Get child nodes for current parent
                parent_node = categories.find_node(parent_id)
                children = categories.find_children(parent_node)

                # Check if child nodes exist
                if not children:
                    output.errors.append(
                        f"No categories found for parent_id {parent_id} in the tree"
                    )
                    end = datetime.now()
                    output.execution_time = (end - start).total_seconds()
                    return output

                # Extract category names and descriptions
                category_list = [
                    f"Category Name: {node.name}, Description: {node.description}"
                    for node in children
                ]
                category_names = [node.name for node in children]

                # Run categorization for this level
                level_output = await self._operator.run(
                    # User parameters
                    text=text,
                    category_list=category_list,
                    with_analysis=with_analysis,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    mode=mode,
                    validator=validator,
                    max_validation_retries=max_validation_retries,
                    # Internal parameters
                    prompt_file="categorize.yaml",
                    output_model=Models.create_dynamic_model(category_names),
                    output_lang=None,
                )

                # Check for errors from operator
                if level_output.errors:
                    output.errors.extend(level_output.errors)
                    end = datetime.now()
                    output.execution_time = (end - start).total_seconds()
                    return output

                # Get the chosen category
                chosen_category = level_output.result

                # Find the corresponding node
                parent_node = categories.find_node(chosen_category)
                if parent_node is None:
                    output.errors.append(
                        f"Category '{chosen_category}' not found in tree after selection"
                    )
                    end = datetime.now()
                    output.execution_time = (end - start).total_seconds()
                    return output

                parent_id = parent_node.node_id
                final_output.append(parent_node.name)

                # Copy analysis/logprobs/process from the last level's output
                output.analysis = level_output.analysis
                output.logprobs = level_output.logprobs
                output.process = level_output.process

            output.result = final_output
            end = datetime.now()
            output.execution_time = (end - start).total_seconds()
            return output

        else:
            output = await self._operator.run(
                # User parameters
                text=text,
                category_list=categories,
                with_analysis=with_analysis,
                user_prompt=user_prompt,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                mode=mode,
                validator=validator,
                max_validation_retries=max_validation_retries,
                # Internal parameters
                prompt_file="categorize.yaml",
                output_model=Models.create_dynamic_model(categories),
                output_lang=None,
            )
            end = datetime.now()
            output.execution_time = (end - start).total_seconds()
            return output

    async def extract_keywords(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        mode: Literal["auto", "threshold", "count"] = "auto",
        number_of_keywords: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Extract salient keywords from text.

        Arguments:
            text: The input text to extract keywords from
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output response
            user_prompt: Additional instructions for keyword extraction
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (list[str]): List of extracted keywords
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="extract_keywords.yaml",
            output_model=Models.ListStrOutput,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

    async def extract_entities(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Perform Named Entity Recognition (NER) over the input text.

        Arguments:
            text: The input text to extract entities from
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output response
            user_prompt: Additional instructions for entity extraction
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (list[dict]): List of entities with 'text' and 'type' keys
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="extract_entities.yaml",
            output_model=Models.ListDictStrStrOutput,
            mode=None,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

    async def is_question(
        self,
        text: str,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Detect if the input is phrased as a question.

        Arguments:
            text: The input text to analyze
            with_analysis: Whether to include detailed reasoning analysis
            user_prompt: Additional instructions for question detection
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (bool): True if text is a question, False otherwise
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="is_question.yaml",
            output_model=Models.BoolOutput,
            mode=None,
            output_lang=None,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

    async def text_to_question(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Generate a single question from the given text.

        Arguments:
            text: The input text to generate a question from
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output question
            user_prompt: Additional instructions for question generation
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (str): The generated question
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="text_to_question.yaml",
            output_model=Models.StrOutput,
            mode=None,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

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
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Merge multiple questions into a single unified question.

        Arguments:
            text: List of questions to merge
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output merged question
            user_prompt: Additional instructions for question merging
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            mode: Merging strategy - 'default' for direct merge, 'reason' for reasoned merge
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (str): The merged question
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        text_combined = ", ".join(text)
        output = await self._operator.run(
            # User parameters
            text=text_combined,
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
            prompt_file="merge_questions.yaml",
            output_model=Models.StrOutput,
            mode=mode,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

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
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Rewrite a text with different modes.

        Arguments:
            text: The input text to rewrite
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output rewritten text
            user_prompt: Additional instructions for rewriting
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            mode: Rewriting mode - 'positive', 'negative', or 'hard_negative'
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (str): The rewritten text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="rewrite.yaml",
            output_model=Models.StrOutput,
            mode=mode,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

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
            output_lang: Language for the output questions
            user_prompt: Additional instructions for question generation
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (list[str]): List of generated questions
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="subject_to_question.yaml",
            output_model=Models.ReasonListStrOutput,
            mode=None,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

    async def summarize(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Summarize the given subject text.

        Arguments:
            text: The input text to summarize
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output summary
            user_prompt: Additional instructions for summarization
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (str): The summary text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="summarize.yaml",
            output_model=Models.StrOutput,
            mode=None,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

    async def translate(
        self,
        text: str,
        target_language: str,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Translate text between languages.

        Arguments:
            text: The input text to translate
            target_language: The target language for translation
            with_analysis: Whether to include detailed reasoning analysis
            user_prompt: Additional instructions for translation
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (str): The translated text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="translate.yaml",
            output_model=Models.StrOutput,
            mode=None,
            output_lang=None,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

    async def detect_entity(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Detects entities in a given text based on the entity_detector.yaml prompt.

        Arguments:
            text: The input text
            with_analysis: Whether to include detailed reasoning analysis
            output_lang: Language for the output summary
            user_prompt: Additional instructions for summarization
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (list[Entity]): The entities
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
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
            prompt_file="detect_entity.yaml",
            output_model=Models.EntityDetectorOutput,
            mode=None,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output

    async def run_custom(
        self,
        prompt: str,
        output_model: Any,
        output_lang: str | None = None,
        temperature: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
        priority: int | None = 0,
    ) -> Models.ToolOutput:
        """
        Custom tool that can do almost anything!

        Arguments:
            text: The user prompt
            output_lang: Language for the output summary
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails
            priority: Task execution priority (if enabled by vLLM and model)

        Returns:
            ToolOutput: Object containing:
                - result (str): The translated text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        start = datetime.now()
        output = await self._operator.run(
            # User paramaeters
            text=prompt,
            output_model=output_model,
            output_model_str=output_model.model_json_schema(),
            output_lang=output_lang,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            validator=validator,
            max_validation_retries=max_validation_retries,
            priority=priority,
            # Internal parameters
            prompt_file="run_custom.yaml",
            user_prompt=None,
            with_analysis=False,
            mode=None,
        )
        end = datetime.now()
        output.execution_time = (end - start).total_seconds()
        return output
