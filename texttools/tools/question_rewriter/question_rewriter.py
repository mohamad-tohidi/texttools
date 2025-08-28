from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    rewritten_question: str


class QuestionRewriter(BaseTool):
    """
    Question Rewriter with optional reasoning step and two modes:
    1. Rewrite with same meaning, different wording.
    2. Rewrite with different meaning, similar wording.
    Outputs JSON with one field: {"rewritten_question": "..."}.
    """

    prompt_file = "question_rewriter.yaml"
    output_model = Output
    use_modes = True

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        with_analysis: bool = False,
        mode: Literal[
            "same_meaning_different_wording_mode",
            "different_meaning_similar_wording_mode",
        ] = "same_meaning_different_wording_mode",
        **kwargs,
    ):
        super().__init__(
            client,
            model=model,
            with_analysis=with_analysis,
            mode=mode,
            **kwargs,
        )

    def rewrite_question(
        self,
        input_text: str,
    ) -> dict[str, str]:
        parsed: Output = self.run(input_text)
        result = self._result_to_dict(parsed.rewritten_question)
        return result
