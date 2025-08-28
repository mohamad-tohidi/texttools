from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from texttools.tools.base_tool import BaseTool


class Output(BaseModel):
    reason: str
    result: Literal[
        "باورهای دینی",
        "اخلاق اسلامی",
        "احکام و فقه",
        "تاریخ اسلام و شخصیت ها",
        "منابع دینی",
        "دین و جامعه/سیاست",
        "عرفان و معنویت",
        "هیچکدام",
    ]


class Categorizer(BaseTool):
    """
    Text categorizer for Islamic studies domain with optional reasoning step.
    Uses an LLM prompt (`categorizer.yaml`) to assign a single `main_tag`
    from a fixed set of categories (e.g., "باورهای دینی", "اخلاق اسلامی", ...).
    Outputs JSON with one field: {"main_tag": "..."}.
    """

    prompt_file = "categorizer.yaml"
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

    def categorize(self, input_text: str) -> dict[str, str]:
        return self.run(input_text)
