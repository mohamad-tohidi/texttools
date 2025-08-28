from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    result: list[dict[str, str]]


class NERExtractor(BaseTool):
    """
    Named Entity Recognition (NER) system with optional reasoning step.
    Outputs JSON with one field: {"entities": [{"text": "...", "type": "..."}, ...]}.
    """

    prompt_file = "ner_extractor.yaml"
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

    def extract_entities(self, input_text: str) -> dict[str, list[dict[str, str]]]:
        return self.run(input_text)
