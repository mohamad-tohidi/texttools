from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from texttools.base_tool import BaseTool


class Output(BaseModel):
    merged_question: str


class QuestionMerger(BaseTool):
    """
    Questions merger for Gemma-style models with one mode for now:
    1. merge the provided questions, preserving all the main points.
    Outputs JSON with a single string field: {"merged_question": "..."}.
    Allows optional extra instructions via `prompt_template`.
    """

    prompt_file = "question_merger.yaml"
    output_model = Output
    use_modes = True

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        use_reason: bool = False,
        mode: Literal["default_mode", "reason_mode"],
        **kwargs,
    ):
        super().__init__(
            client,
            model=model,
            use_reason=use_reason,
            mode=mode,
            **kwargs,
        )

    def merge(
        self,
        questions: list[str],
    ) -> dict[str, str]:
        input_text = ", ".join(questions)
        parsed: Output = self.run(input_text)
        result = {"merged_question": parsed.merged_question}
        self._dispatch(result)
        return result
