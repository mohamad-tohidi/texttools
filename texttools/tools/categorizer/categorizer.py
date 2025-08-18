from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from ...base_tool import BaseTool


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

    def categorize(self, text: str) -> dict:
        parsed: Output = self.run(text)
        results = {"main_tag": parsed.main_tag}
        self._dispatch(results)
        return results
