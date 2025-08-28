from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    generated_question: str


class QuestionGenerator(BaseTool):
    """
    Question Generator with optional reasoning step.
    Outputs JSON with one field: {"generated_question": "..."}.
    """

    prompt_file = "question_generator.yaml"
    output_model = Output

    def __init__(
        self, client: OpenAI, *, model: str, with_analysis: bool = False, **kwargs
    ):
        super().__init__(
            client,
            model=model,
            with_analysis=with_analysis,
            **kwargs,
        )

    def generate_question(self, input_text: str) -> dict[str, str]:
        parsed: Output = self.run(input_text)
        result = self._result_to_dict(input_text, parsed.generated_question)
        self._dispatch(result)
        return result
