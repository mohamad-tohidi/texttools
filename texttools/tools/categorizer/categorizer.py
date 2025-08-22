from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from texttools.base_tool import BaseTool


class Output(BaseModel):
    reason: str
    main_tag: Literal[
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
    Text categorizer for Islamic studies domain.
    Uses an LLM prompt (`categorizer.yaml`) to assign a single `main_tag`
    from a fixed set of categories (e.g., "باورهای دینی", "اخلاق اسلامی", ...).
    Outputs JSON with one field: {"main_tag": "<category>"}.
    Optionally includes reasoning when `use_reason=True`.
    """

    prompt_file = "categorizer.yaml"
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

    def categorize(self, text: str) -> dict[str, str]:
        parsed: Output = self.run(text)
        result = {"main_tag": parsed.main_tag}
        self._dispatch(result)
        return result
