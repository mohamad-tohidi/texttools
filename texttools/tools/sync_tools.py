from typing import Literal, Any, Callable

from openai import OpenAI

from texttools.tools.internals.sync_operator import Operator
import texttools.tools.internals.output_models as OM


class TheTool:
    """
    Each method configures the operator with a specific YAML prompt,
    output schema, and flags, then delegates execution to `operator.run()`.

    Usage:
        client = OpenAI(...)
        tool = TheTool(client, model="model-name")
        result = tool.categorize("text ...", with_analysis=True)
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ):
        self._operator = Operator(client=client, model=model)

    def categorize(
        self,
        text: str,
        category: OM.CategoryTree,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
    ) -> OM.ToolOutput:
        """
        Categorize a text into a single Islamic studies domain category.

        Arguments:
            text: The input text to categorize
            with_analysis: Whether to include detailed reasoning analysis
            user_prompt: Additional instructions for the categorization
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            logprobs: Whether to return token probability information
            top_logprobs: Number of top token alternatives to return if logprobs enabled
            validator: Custom validation function to validate the output
            max_validation_retries: Maximum number of retry attempts if validation fails

        Returns:
            ToolOutput: Object containing:
                - result (str): The assigned Islamic studies category
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            validator=validator,
            max_validation_retries=max_validation_retries,
            # Internal parameters
            prompt_file="categorizer.yaml",
            output_model=OM.CategorizerOutput,
            mode=None,
            output_lang=None,
            category=category,
        )

    def extract_keywords(
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
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (list[str]): List of extracted keywords
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
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
            # Internal parameters
            prompt_file="extract_keywords.yaml",
            output_model=OM.ListStrOutput,
            mode=None,
        )

    def extract_entities(
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
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (list[dict]): List of entities with 'text' and 'type' keys
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
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
            # Internal parameters
            prompt_file="extract_entities.yaml",
            output_model=OM.ListDictStrStrOutput,
            mode=None,
        )

    def is_question(
        self,
        text: str,
        with_analysis: bool = False,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (bool): True if text is a question, False otherwise
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            validator=validator,
            max_validation_retries=max_validation_retries,
            # Internal parameters
            prompt_file="is_question.yaml",
            output_model=OM.BoolOutput,
            mode=None,
            output_lang=None,
        )

    def text_to_question(
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
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (str): The generated question
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
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
            # Internal parameters
            prompt_file="text_to_question.yaml",
            output_model=OM.StrOutput,
            mode=None,
        )

    def merge_questions(
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
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (str): The merged question
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        text = ", ".join(text)
        return self._operator.run(
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
            # Internal parameters
            prompt_file="merge_questions.yaml",
            output_model=OM.StrOutput,
            mode=mode,
        )

    def rewrite(
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
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (str): The rewritten text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
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
            # Internal parameters
            prompt_file="rewrite.yaml",
            output_model=OM.StrOutput,
            mode=mode,
        )

    def subject_to_question(
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
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (list[str]): List of generated questions
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
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
            # Internal parameters
            prompt_file="subject_to_question.yaml",
            output_model=OM.ReasonListStrOutput,
            mode=None,
        )

    def summarize(
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
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (str): The summary text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
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
            # Internal parameters
            prompt_file="summarize.yaml",
            output_model=OM.StrOutput,
            mode=None,
        )

    def translate(
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
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (str): The translated text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
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
            # Internal parameters
            prompt_file="translate.yaml",
            output_model=OM.StrOutput,
            mode=None,
            output_lang=None,
        )

    def run_custom(
        self,
        prompt: str,
        output_model: Any,
        output_lang: str | None = None,
        temperature: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        validator: Callable[[Any], bool] | None = None,
        max_validation_retries: int | None = None,
    ) -> OM.ToolOutput:
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

        Returns:
            ToolOutput: Object containing:
                - result (str): The translated text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
                - errors (list(str) | None): Errors occured during tool call
        """
        return self._operator.run(
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
            # Internal parameters
            prompt_file="run_custom.yaml",
            user_prompt=None,
            with_analysis=False,
            mode=None,
        )

    def entity_detector(
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
    ) -> OM.ToolOutput:
        """
        Detects entities in a given text based on the entity_detector.yaml prompt.
        """
        return self._operator.run(
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            validator=validator,
            max_validation_retries=max_validation_retries,
            prompt_file="entity_detector.yaml",
            output_model=OM.EntityDetectorOutput,
            mode=None,
        )
