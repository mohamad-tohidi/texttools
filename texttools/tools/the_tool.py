from typing import Optional, Literal

from openai import OpenAI

from texttools.tools.internals.operator import Operator
import texttools.tools.internals.output_models as OutputModels


class TheTool:
    """
    High-level interface exposing specialized text tools for.

    Each method configures the operator with a specific YAML prompt,
    output schema, and flags, then delegates execution to `operator.run()`.

    Supported capabilities:
    - categorize: assign a text to one of several Islamic categories.
    - extract_keywords: produce a keyword list from text.
    - extract_entities: simple NER (name/type pairs).
    - detect_question: binary check whether input is a question.
    - generate_question_from_text: produce a new question from a text.
    - merge_questions: combine multiple questions (default/reason modes).
    - rewrite_question: rephrase questions (same meaning/different wording, or vice versa).
    - generate_questions_from_subject: generate multiple questions given a subject.
    - summarize: produce a concise summary of a subject.
    - translate: translate text between languages.

    Usage pattern:
        client = OpenAI(...)
        tool = TheTool(client, model="gemma-3")
        result = tool.categorize("متن ورودی ...", with_analysis=True)
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str = "google/gemma-3n-e4b-it",
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: bool = False,
        temperature: float = 0.0,
        logprobs: bool = False,
        top_logprobs: int = 3,
    ):
        # Initialize Operator
        self.operator = Operator(client=client)

        # Initialize default values
        self.model = model
        self.user_prompt = user_prompt
        self.output_lang = output_lang
        self.with_analysis = with_analysis
        self.temperature = temperature
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

    def categorize(
        self,
        text: str,
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
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

        results = self.operator.run(
            # Internal parameters
            prompt_file="categorizer.yaml",
            output_model=OutputModels.CategorizerOutput,
            resp_format="parse",
            # User parameters
            text=text,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            output_lang=self.output_lang if output_lang is None else output_lang,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def extract_keywords(
        self,
        text: str,
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, list[str]]:
        """
        Extract salient keywords from text.

        Args:
            text: Input string to analyze.
            with_analysis: Whether to run an extra LLM reasoning step.

        Returns:
            {"result": [<keyword1>, <keyword2>, ...]}
        """
        results = self.operator.run(
            # Internal parameters
            prompt_file="keyword_extractor.yaml",
            output_model=OutputModels.ListStrOutput,
            resp_format="parse",
            # User parameters
            text=text,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            output_lang=self.output_lang if output_lang is None else output_lang,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def extract_entities(
        self,
        text: str,
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, list[dict[str, str]]]:
        """
        Perform Named Entity Recognition (NER) over the input text.

        Args:
            text: Input string.
            with_analysis: Whether to run an extra LLM reasoning step.

        Returns:
            {"result": [{"text": <entity>, "type": <entity_type>}, ...]}
        """
        results = self.operator.run(
            # Internal parameters
            prompt_file="ner_extractor.yaml",
            output_model=OutputModels.ListDictStrStrOutput,
            resp_format="parse",
            # User parameters
            text=text,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            output_lang=self.output_lang if output_lang is None else output_lang,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def detect_question(
        self,
        text: str,
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, bool]:
        """
        Detect if the input is phrased as a question.

        Args:
            question: Input string to evaluate.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": "true"} or {"result": "false"}
        """
        results = self.operator.run(
            # Internal parameters
            prompt_file="question_detector.yaml",
            output_model=OutputModels.BoolOutput,
            resp_format="parse",
            output_lang=False,
            # User parameters
            text=text,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def generate_question_from_text(
        self,
        text: str,
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, str]:
        """
        Generate a single question from the given text.

        Args:
            text: Source text to derive a question from.
            with_analysis: Whether to use analysis before generation.

        Returns:
            {"result": <generated_question>}
        """
        results = self.operator.run(
            # Internal parameters
            prompt_file="question_generator.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            # User parameters
            text=text,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            output_lang=self.output_lang if output_lang is None else output_lang,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def merge_questions(
        self,
        questions: list[str],
        mode: Literal["default", "reason"] = "default",
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, str]:
        """
        Merge multiple questions into a single unified question.

        Args:
            questions: List of question strings.
            mode: Merge strategy:
                - "default": simple merging.
                - "reason": merging with reasoning explanation.
            with_analysis: Whether to use an analysis step.

        Returns:
            {"result": <merged_question>}
        """
        text = ", ".join(questions)

        results = self.operator.run(
            # Internal parameters
            prompt_file="question_merger.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            # User parameters
            text=text,
            mode=mode,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            output_lang=self.output_lang if output_lang is None else output_lang,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def rewrite_question(
        self,
        text: str,
        mode: Literal[
            "same_meaning_different_wording",
            "different_meaning_similar_wording",
        ] = "same_meaning_different_wording",
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, str]:
        """
        Rewrite a question with different wording or meaning.

        Args:
            question: Input question to rewrite.
            mode: Rewrite strategy:
                - "same_meaning_different_wording": keep meaning, change words.
                - "different_meaning_similar_wording": alter meaning, preserve wording style.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": <rewritten_question>}
        """
        results = self.operator.run(
            # Internal parameters
            prompt_file="question_rewriter.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            # User parameters
            text=text,
            mode=mode,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            output_lang=self.output_lang if output_lang is None else output_lang,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def generate_questions_from_subject(
        self,
        text: str,
        number_of_questions: int,
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, list[str]]:
        """
        Generate a list of questions about a subject.

        Args:
            subject: Topic of interest.
            number_of_questions: Number of questions to produce.
            language: Target language for generated questions.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": [<question1>, <question2>, ...]}
        """
        results = self.operator.run(
            # Internal parameters
            prompt_file="subject_question_generator.yaml",
            output_model=OutputModels.ReasonListStrOutput,
            resp_format="parse",
            # User parameters
            text=text,
            number_of_questions=number_of_questions,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            output_lang=self.output_lang if output_lang is None else output_lang,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def summarize(
        self,
        text: str,
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_lang: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, str]:
        """
        Summarize the given subject text.

        Args:
            subject: Input text to summarize.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": <summary>}
        """
        results = self.operator.run(
            # Internal parameters
            prompt_file="summarizer.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            # User paramaeters
            text=text,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            output_lang=self.output_lang if output_lang is None else output_lang,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results

    def translate(
        self,
        text: str,
        target_language: str,
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        with_analysis: Optional[bool] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> dict[str, str]:
        """
        Translate text between languages.

        Args:
            text: Input string to translate.
            target_language: Language code or name to translate into.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": <translated_text>}
        """
        results = self.operator.run(
            # Internal parameters
            prompt_file="translator.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            output_lang=False,
            # User parameters
            text=text,
            target_language=target_language,
            model=self.model if model is None else model,
            user_prompt=self.user_prompt if user_prompt is None else user_prompt,
            with_analysis=self.with_analysis
            if with_analysis is None
            else with_analysis,
            temperature=self.temperature if temperature is None else temperature,
            logprobs=self.logprobs if logprobs is None else logprobs,
            top_logprobs=self.top_logprobs if top_logprobs is None else top_logprobs,
        )

        return results
