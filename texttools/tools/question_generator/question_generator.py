from openai import OpenAI
from pydantic import BaseModel

from texttools.base_tool import BaseTool


class Output(BaseModel):
    generated_question: str


class QuestionGenerator(BaseTool):
    """
    Question Generator for Gemma-style models with optional reasoning step.
    Outputs JSON with a single string field: {"generated_question": "..."}.
    Optionally includes reasoning when `use_reason=True`.
    """

    prompt_file = "question_generator.yaml"
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

    def generate_question(self, text: str) -> str:
        parsed: Output = self.run(text)
        result = {"generated_question": parsed.generated_question}
        self._dispatch(result)
        return result
