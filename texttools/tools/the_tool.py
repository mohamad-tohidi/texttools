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
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Categorize a text into a single Islamic studies domain category.

        Returns:
            ToolOutput: Object containing:
                - result (str): The assigned Islamic studies category
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
        """
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            validator=validator,
            # Internal parameters
            prompt_file="categorizer.yaml",
            output_model=OutputModels.CategorizerOutput,
            resp_format="parse",
            mode=None,
            output_lang=None,
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
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Extract salient keywords from text.

        Returns:
            ToolOutput: Object containing:
                - result (list[str]): List of extracted keywords
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
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
            validator=validator,
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
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Perform Named Entity Recognition (NER) over the input text.

        Returns:
            ToolOutput: Object containing:
                - result (list[dict]): List of entities with 'text' and 'type' keys
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
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
            validator=validator,
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
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Detect if the input is phrased as a question.

        Returns:
            ToolOutput: Object containing:
                - result (bool): True if text is a question, False otherwise
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
        """
        return self.operator.run(
            # User parameters
            text=text,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            validator=validator,
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
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Generate a single question from the given text.

        Returns:
            ToolOutput: Object containing:
                - result (str): The generated question
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
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
            validator=validator,
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
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Merge multiple questions into a single unified question.

        Returns:
            ToolOutput: Object containing:
                - result (str): The merged question
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
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
            validator=validator,
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
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Rewrite a text with different modes.

        Returns:
            ToolOutput: Object containing:
                - result (str): The rewritten text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
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
            validator=validator,
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
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Generate a list of questions about a subject.

        Returns:
            ToolOutput: Object containing:
                - result (list[str]): List of generated questions
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
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
            validator=validator,
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
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Summarize the given subject text.

        Returns:
            ToolOutput: Object containing:
                - result (str): The summary text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
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
            validator=validator,
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
        user_prompt: str | None = None,
        temperature: float | None = 0.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        validator: Any | None = None,
    ) -> OutputModels.ToolOutput:
        """
        Translate text between languages.

        Returns:
            ToolOutput: Object containing:
                - result (str): The translated text
                - logprobs (list | None): Probability data if logprobs enabled
                - analysis (str | None): Detailed reasoning if with_analysis enabled
        """
        return self.operator.run(
            # User parameters
            text=text,
            target_language=target_language,
            with_analysis=with_analysis,
            user_prompt=user_prompt,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            validator=validator,
            # Internal parameters
            prompt_file="translate.yaml",
            output_model=OutputModels.StrOutput,
            resp_format="parse",
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
    ) -> OutputModels.ToolOutput:
        """
        Custom tool that can do almost anything!

        Returns:
            ToolOutput: Object with fields:
                - result (str): The output result
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
            validator=None,
        )
