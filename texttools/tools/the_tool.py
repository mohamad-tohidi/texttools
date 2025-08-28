from typing import Literal, Any

from openai import OpenAI

from texttools.tools.operator import Operator
import texttools.tools.output_models as OutputModels


class TheTool:
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
        Text categorizer for Islamic studies domain with optional analyzing step.
        Uses an LLM prompt (`categorizer.yaml`) to assign a single `main_tag`
        from a fixed set of categories (e.g., "باورهای دینی", "اخلاق اسلامی", ...).
        Outputs JSON with one field: {"result": "..."}.
        """
        self.operator.PROMPT_FILE = "categorizer.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.CategorizerOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(text)
        return results

    def extract_keywords(
        self, text: str, with_analysis: bool = False
    ) -> dict[str, str]:
        """
        Keyword extractor for with optional analyzing step.
        Outputs JSON with one field: {"result": ["keyword1", "keyword2", ...]}.
        """
        self.operator.PROMPT_FILE = "keyword_extractor.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.ListStrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(text)
        return results

    def extract_entities(
        self, text: str, with_analysis: bool = False
    ) -> dict[str, str]:
        """
        Named Entity Recognition (NER) system with optional analyzing step.
        Outputs JSON with one field: {"result": [{"text": "...", "type": "..."}, ...]}.
        """
        self.operator.PROMPT_FILE = "ner_extractor.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.ListDictStrStrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(text)
        return results

    def detect_question(
        self, question: str, with_analysis: bool = False
    ) -> dict[str, str]:
        """
        Binary question detector with optional analyzing step..
        Outputs JSON with one field: {"result": true/false}.
        """
        self.operator.PROMPT_FILE = "question_detector.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.StrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(question)
        return results

    def generate_question(
        self, text: str, with_analysis: bool = False
    ) -> dict[str, str]:
        """
        Question Generator with optional analyzing step.
        Outputs JSON with one field: {"result": "..."}.
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
        Questions merger with optional analyzing step and two modes:
        1. Default mode
        2. Reason mode
        Outputs JSON with one field: {"result": "..."}.
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
        Question Rewriter with optional analyzing step and two modes:
        1. Rewrite with same meaning, different wording.
        2. Rewrite with different meaning, similar wording.
        Outputs JSON with one field: {"result": "..."}.
        """

        self.operator.PROMPT_FILE = "question_rewriter.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.StrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = True
        self.operator.MODE = mode

        results = self.operator.run(question)
        return results

    def generate_subject_question(
        self,
        subject: str,
        number_of_questions: int,
        language: str,
        with_analysis: bool = False,
    ) -> dict[str, str]:
        """
        Subject question generator with optional analyzing step.
        Outputs JSON with one field: {"result": "..."}.
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
        Summarizer with optional analyzing step.
        Outputs JSON with one field: {"result": "..."}.
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
        source_language: str,
        with_analysis: bool = False,
    ) -> dict[str, str]:
        """
        Translator with optional analyzing step.
        Outputs JSON with one field: {"result": "..."}.
        """
        self.operator.PROMPT_FILE = "translator.yaml"
        self.operator.OUTPUT_MODEL = OutputModels.StrOutput
        self.operator.WITH_ANALYSIS = with_analysis
        self.operator.USE_MODES = False

        results = self.operator.run(
            text,
            target_language=target_language,
            source_language=source_language,
        )
        return results
