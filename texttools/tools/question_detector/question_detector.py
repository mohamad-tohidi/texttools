from openai import OpenAI
from pydantic import BaseModel

from texttools.base_tool import BaseTool


class Output(BaseModel):
    is_question: bool


class QuestionDetector(BaseTool):
    """
    Simplified binary question detector.
    Outputs JSON with a single boolean field: {"is_question": true|false}.
    Optionally includes reasoning when `use_reason=True`.
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
        result = {"is_question": parsed.is_question}
        self._dispatch({"question": text, "result": result})
        return result
