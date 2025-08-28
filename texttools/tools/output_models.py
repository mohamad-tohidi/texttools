from typing import Literal

from pydantic import BaseModel


class StrOutput(BaseModel):
    result: str


class ListStrOutput(BaseModel):
    result: list[str]


class ListDictStrStrOutput(BaseModel):
    result: list[dict[str, str]]


class ReasonListStrOutput(BaseModel):
    reason: str
    result: list[str]


class CategorizerOutput(BaseModel):
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
