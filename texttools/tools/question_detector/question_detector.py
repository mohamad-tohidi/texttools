from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    is_question: bool


class QuestionDetector(BaseTool):
    """
    Binary question detector with optional reasoning step..
    Outputs JSON with one field: {"is_question": true/false}.
    """

    prompt_file = "question_detector.yaml"
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

    def detect(self, input_text: str) -> dict[str, str]:
        parsed: Output = self.run(input_text)
        result = self._result_to_dict(input_text, parsed.is_question)
        return result
