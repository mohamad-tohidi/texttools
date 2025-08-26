from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    is_question: bool


class QuestionDetector(BaseTool):
    """
    Binary question detector with optional reasoning step..
    Outputs JSON with one field: {"is_question": true|false}.
    """

    prompt_file = "question_detector.yaml"
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

    def detect(self, text: str) -> dict[str, str]:
        parsed: Output = self.run(text)
        result = self._result_to_dict(parsed.is_question)
        self._dispatch({"question": text, "result": result})
        return result
