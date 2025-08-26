from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    translation: str


class Translator(BaseTool):
    """
    Translator with optional reasoning step.
    Outputs JSON with one field: {"translation": "..."}.
    """

    prompt_file = "translator.yaml"
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

    def translate(
        self, text: str, target_language: str, source_language: str
    ) -> dict[str, str]:
        parsed: Output = self.run(
            text,
            target_language=target_language,
            source_language=source_language,
        )
        result = self._result_to_dict(parsed.translation)
        self._dispatch(result)
        return result
