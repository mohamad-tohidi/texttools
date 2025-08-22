from openai import OpenAI
from pydantic import BaseModel

from texttools.base_tool import BaseTool


class Output(BaseModel):
    summary: str


class Summarizer(BaseTool):
    """
    Summarizer.
    Outputs JSON with a single string field: {"summary": "..."}.
    Optionally includes reasoning when `use_reason=True`.
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
