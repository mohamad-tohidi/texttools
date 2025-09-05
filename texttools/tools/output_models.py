from typing import Literal

from pydantic import BaseModel


class StrOutput(BaseModel):
    """
    Output model for a single string result.
    """

    result: str


class ListStrOutput(BaseModel):
    """
    Output model for a list of strings result.
    """

    result: list[str]


class ListDictStrStrOutput(BaseModel):
    """
    Output model for a list of dictionaries with string key-value pairs.
    """

    result: list[dict[str, str]]


class ReasonListStrOutput(BaseModel):
    """
    Output model containing a reasoning string followed by a list of strings.
    """

    reason: str
    result: list[str]


class CategorizerOutput(BaseModel):
    """
    Output model for categorization with reasoning and a predefined category result.
    """

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
