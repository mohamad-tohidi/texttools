from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


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
        self, client: OpenAI, *, model: str, with_analysis: bool = False, **kwargs
    ):
        super().__init__(
            client,
            model=model,
            with_analysis=with_analysis,
            **kwargs,
        )

    def summarize(self, input_text: str) -> dict[str, str]:
        parsed: Output = self.run(input_text)
        result = self._build_results_dict(parsed.summary)
        return result
