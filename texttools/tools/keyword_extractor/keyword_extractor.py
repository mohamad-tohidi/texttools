from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    keywords: list[str]


class KeywordExtractor(BaseTool):
    """
    Keyword extractor for with optional reasoning step.
    Outputs JSON with one field: {"keywords": ["keyword1", "keyword2", ...]}.
    """

    prompt_file = "keyword_extractor.yaml"
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

    def extract_keywords(self, text: str) -> dict[str, list[str]]:
        parsed: Output = self.run(text)
        result = {"keywords": parsed.keywords}
        self._dispatch(result)
        return result
