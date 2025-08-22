from openai import OpenAI
from pydantic import BaseModel

from texttools.base_tool import BaseTool


class Output(BaseModel):
    entities: list[dict[str, str]]


class NERExtractor(BaseTool):
    """
    Named Entity Recognition (NER) system.
    Outputs JSON with a single array field: {"entities": [{"text": "...", "type": "..."}, ...]}.
    Optionally includes reasoning when `use_reason=True`
    """

    prompt_file = "ner_extractor.yaml"
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

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        parsed: Output = self.run(text)
        result = {"entities": parsed.entities}
        self._dispatch(result)
        return result
