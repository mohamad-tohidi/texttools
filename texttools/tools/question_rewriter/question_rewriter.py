from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from texttools.base_tool import BaseTool


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
        use_reason: bool = False,
        mode: Literal[
            "same_meaning_different_wording_mode",
            "different_meaning_similar_wording_mode",
        ],
        **kwargs,
    ):
        super().__init__(
            client,
            model=model,
            use_reason=use_reason,
            mode=mode,
            **kwargs,
        )

    def rewrite_question(
        self,
        question: str,
    ) -> dict[str, str]:
        parsed: Output = self.run(question)
        result = {"rewritten_question": parsed.rewritten_question}
        self._dispatch(result)
        return result
