from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    merged_question: str


class QuestionMerger(BaseTool):
    """
    Questions merger with optional reasoning step and two modes:
    1. Default mode
    2. Reason mode
    Outputs JSON with one field: {"merged_question": "..."}.
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
        mode: Literal["default_mode", "reason_mode"] = "default_mode",
        **kwargs,
    ):
        super().__init__(
            client,
            model=model,
            use_reason=use_reason,
            mode=mode,
            **kwargs,
        )

    def merge_questions(
        self,
        questions: list[str],
    ) -> dict[str, str]:
        input_text = ", ".join(questions)
        parsed: Output = self.run(input_text)
        result = self._result_to_dict(input_text, parsed.merged_question)
        self._dispatch(result)
        return result
