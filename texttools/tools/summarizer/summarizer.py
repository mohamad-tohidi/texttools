from openai import OpenAI
from pydantic import BaseModel

from texttools.base_tool import BaseTool


class Output(BaseModel):
    summary: str


class Summarizer(BaseTool):
    """
    Summarizer with optional reasoning step.
    Outputs JSON with one field: {"summary": "..."}.
    """

    prompt_file = "summarizer.yaml"
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

    def summarize(self, text: str) -> dict[str, str]:
        parsed: Output = self.run(text)
        result = {"summary": parsed.summary}
        self._dispatch(result)
        return result
