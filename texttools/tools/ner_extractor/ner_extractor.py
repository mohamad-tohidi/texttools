from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    entities: list[dict[str, str]]


class NERExtractor(BaseTool):
    """
    Named Entity Recognition (NER) system with optional reasoning step.
    Outputs JSON with one field: {"entities": [{"text": "...", "type": "..."}, ...]}.
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

    def extract_entities(self, text: str) -> dict[str, list[dict[str, str]]]:
        parsed: Output = self.run(text)
        result = self._result_to_dict(parsed.entities)
        self._dispatch(result)
        return result
