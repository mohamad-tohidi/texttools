from typing import Literal, Any

from openai import OpenAI

from texttools.tools.internals.operator import Operator
import texttools.tools.internals.output_models as OutputModels


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
        self.operator = Operator(client=client, model=model)

    def categorize(
        self,
        text: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
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
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="categorizer.yaml",
            output_model=OutputModels.CategorizerOutput,
            resp_format="parse",
            mode=None,
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
    ) -> dict[str, list[str]]:
        """
        Extract salient keywords from text.

        Args:
            text: Input string to analyze.
            with_analysis: Whether to run an extra LLM reasoning step.

        Returns:
            {"result": [<keyword1>, <keyword2>, ...]}
        """
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="extract_keywords.yaml",
            output_model=OutputModels.ListStrOutput,
            resp_format="parse",
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
    ) -> dict[str, list[dict[str, str]]]:
        """
        Perform Named Entity Recognition (NER) over the input text.

        Args:
            text: Input string.
            with_analysis: Whether to run an extra LLM reasoning step.

        Returns:
            {"result": [{"text": <entity>, "type": <entity_type>}, ...]}
        """
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="extract_entities.yaml",
            output_model=OutputModels.ListDictStrStrOutput,
            resp_format="parse",
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
    ) -> dict[str, bool]:
        """
        Detect if the input is phrased as a question.

        Args:
            question: Input string to evaluate.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": "true"} or {"result": "false"}
        """
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="is_question.yaml",
            output_model=OutputModels.BoolOutput,
            resp_format="parse",
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
    ) -> dict[str, str]:
        """
        Generate a single question from the given text.

        Args:
            text: Source text to derive a question from.
            with_analysis: Whether to use analysis before generation.

        Returns:
            {"result": <generated_question>}
        """
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="text_to_question.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
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
        text = ", ".join(text)
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="merge_questions.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
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
    ) -> dict[str, str]:
        """
        Rewrite a question with different wording or meaning.

        Args:
            question: Input question to rewrite.
            mode: Rewrite strategy:
                - "positive": keep meaning, change words.
                - "negative": alter meaning, preserve wording style.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": <rewritten_question>}
        """
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="rewrite.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
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
        return self.operator.run(
            # User parameters
            text=text,
            number_of_questions=number_of_questions,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="subject_to_question.yaml",
            output_model=OutputModels.ReasonListStrOutput,
            resp_format="parse",
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
    ) -> dict[str, str]:
        """
        Summarize the given subject text.

        Args:
            subject: Input text to summarize.
            with_analysis: Whether to include an analysis step.

        Returns:
            {"result": <summary>}
        """
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="summarize.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            mode=None,
        )

    def translate(
        self,
        text: str,
        target_language: str,
        with_analysis: bool = False,
        output_lang: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
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
        return self.operator.run(
            # User parameters
            text=text,
            target_language=target_language,
            with_analysis=with_analysis,
            output_lang=output_lang,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="translate.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
            mode=None,
        )

    def run_custom(
        self,
        prompt: str,
        output_model: Any,
        output_lang: str | None = None,
        temperature: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> dict[str, Any]:
        """
        Custom tool that can do almost anything!

        Args:
            prompt: Custom prompt.
            output_model: Custom BaseModel output model.

        Returns:
            {"result": <Any>}
        """
        return self.operator.run(
            # User paramaeters
            text=prompt,
            output_model=output_model,
            output_model_str=output_model.model_json_schema(),
            output_lang=output_lang,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            # Internal parameters
            prompt_file="run_custom.yaml",
            resp_format="parse",
            user_prompt=None,
            with_analysis=False,
            mode=None,
        )
