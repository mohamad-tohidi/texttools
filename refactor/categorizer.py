# texttools/tools/gemma_categorizer.py
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from base_tool import BaseTool
from texttools.formatter.gemma3_fromatter import Gemma3Formatter


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
            chat_formatter=Gemma3Formatter(),
            **kwargs,
        )

    def categorize(self, text: str) -> dict:
        parsed: Output = self.run(text)
        self._dispatch(results={"main_tag": parsed.main_tag})
        return parsed.dict()
