from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    reasoning_summary: str
    generated_questions: list[str]


class SubjectQuestionGenerator(BaseTool):
    """
    Question Generator with optional reasoning step.
    Outputs JSON with one field: {"generated_questions": "..."}.
    """

    prompt_file = "subject_question_generator.yaml"
    output_model = Output

    def __init__(
        self, client: OpenAI, *, model: str, use_reason: bool = False, **kwargs
    ):
        super().__init__(
            client,
            model=model,
            use_reason=use_reason,
            **kwargs,
        )

    def generate_question(
        self, input_text: str, number_of_questions: int, language: str
    ) -> dict[str, list[str]]:
        parsed: Output = self.run(
            input_text,
            number_of_questions=number_of_questions,
            language=language,
        )
        result = self._result_to_dict(input_text, parsed.generated_questions)
        self._dispatch(result)
        return result
