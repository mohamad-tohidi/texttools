from typing import Literal, Any

from openai import OpenAI

from texttools.tools.operator import Operator
import texttools.tools.output_models as OutputModels


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
        model: str,
        temperature: float = 0.0,
        **client_kwargs: Any,
    ):
        self.operator = Operator(
            client=client,
            model=model,
            temperature=temperature,
            **client_kwargs,
        )

    def categorize(self, text: str, with_analysis: bool = False) -> dict[str, str]:
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
        self.operator.PROMPT_FILE = "categorizer.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.CategorizerOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(text)
        return results

    def extract_keywords(
        self, text: str, with_analysis: bool = False
    ) -> dict[str, list[str]]:
        """
        Extract salient keywords from text.

        Args:
            text: Input string to analyze.
            with_analysis: Whether to run an extra LLM reasoning step.

        Returns:
            {"result": [<keyword1>, <keyword2>, ...]}
        """
        self.operator.PROMPT_FILE = "keyword_extractor.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.ListStrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(text)
        return results

    def extract_entities(
        self, text: str, with_analysis: bool = False
    ) -> dict[str, list[dict[str, str]]]:
        """
        Perform Named Entity Recognition (NER) over the input text.

        Args:
            text: Input string.
            with_analysis: Whether to run an extra LLM reasoning step.

        Returns:
            {"result": [{"text": <entity>, "type": <entity_type>}, ...]}
        """
        self.operator.PROMPT_FILE = "ner_extractor.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.ListDictStrStrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(text)
        return results

    def detect_question(
        self, question: str, with_analysis: bool = False
    ) -> dict[str, bool]:
        """
        Detect if the input is phrased as a question.

        Args:
            question: Input string to evaluate.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": "true"} or {"result": "false"}
        """
        self.operator.PROMPT_FILE = "question_detector.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.BoolOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(question)
        return results

    def generate_question_from_text(
        self, text: str, with_analysis: bool = False
    ) -> dict[str, str]:
        """
        Generate a single question from the given text.

        Args:
            text: Source text to derive a question from.
            with_analysis: Whether to use analysis before generation.

        Returns:
            {"result": <generated_question>}
        """
        self.operator.PROMPT_FILE = "question_generator.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.StrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(text)
        return results

    def merge_questions(
        self,
        questions: list[str],
        mode: Literal["default_mode", "reason_mode"] = "default_mode",
        with_analysis: bool = False,
    ) -> dict[str, str]:
        """
        Merge multiple questions into a single unified question.

        Args:
            questions: List of question strings.
            mode: Merge strategy:
                - "default_mode": simple merging.
                - "reason_mode": merging with reasoning explanation.
            with_analysis: Whether to use an analysis step.

        Returns:
            {"result": <merged_question>}
        """
        question_str = ", ".join(questions)

        self.operator.PROMPT_FILE = "question_merger.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.StrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = True
        self.operator.MODE = mode

        results = self.operator.run(question_str)
        return results

    def rewrite_question(
        self,
        question: str,
        mode: Literal[
            "same_meaning_different_wording_mode",
            "different_meaning_similar_wording_mode",
        ] = "same_meaning_different_wording_mode",
        with_analysis: bool = False,
    ) -> dict[str, str]:
        """
        Rewrite a question with different wording or meaning.

        Args:
            question: Input question to rewrite.
            mode: Rewrite strategy:
                - "same_meaning_different_wording_mode": keep meaning, change words.
                - "different_meaning_similar_wording_mode": alter meaning, preserve wording style.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": <rewritten_question>}
        """
        self.operator.PROMPT_FILE = "question_rewriter.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.StrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = True
        self.operator.MODE = mode

        results = self.operator.run(question)
        return results

    def generate_questions_from_subject(
        self,
        subject: str,
        number_of_questions: int,
        language: str = "English",
        with_analysis: bool = False,
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
        self.operator.PROMPT_FILE = "subject_question_generator.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.ReasonListStrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(
            subject,
            number_of_questions=number_of_questions,
            language=language,
        )
        return results

    def summarize(self, subject: str, with_analysis: bool = False) -> dict[str, str]:
        """
        Summarize the given subject text.

        Args:
            subject: Input text to summarize.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": <summary>}
        """
        self.operator.PROMPT_FILE = "summarizer.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.StrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(subject)
        return results

    def translate(
        self,
        text: str,
        target_language: str,
        with_analysis: bool = False,
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
        self.operator.PROMPT_FILE = "translator.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.StrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(
            text,
            target_language=target_language,
        )
        return results
