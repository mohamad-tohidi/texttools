from typing import Literal, Any

from pydantic import BaseModel, Field


class ToolOutput(BaseModel):
    result: Any = None
    analysis: str = ""
    logprobs: list[dict[str, Any]] = []
    errors: list[str] = []

    def __repr__(self):
        return f"ToolOutput(Result Type = {type(self.result)}, Result Value = {self.result}, Analysis = {self.analysis}, Logprobs = {self.logprobs}, Errors = {self.errors}"


class StrOutput(BaseModel):
    result: str = Field(..., description="The output string")


class BoolOutput(BaseModel):
    result: bool = Field(
        ..., description="Boolean indicating the output state", example=True
    )


class ListStrOutput(BaseModel):
    result: list[str] = Field(
        ..., description="The output list of strings", example=["text_1", "text_2"]
    )


class ListDictStrStrOutput(BaseModel):
    result: list[dict[str, str]] = Field(
        ...,
        description="List of dictionaries containing string key-value pairs",
        example=[{"text": "Mohammad", "type": "PER"}],
    )


class ReasonListStrOutput(BaseModel):
    reason: str = Field(..., description="Thinking process that led to the output")
    result: list[str] = Field(..., description="The output list of strings")


class CategorizerOutput(BaseModel):
    reason: str = Field(
        ..., description="Explanation of why the input belongs to the category"
    )
    result: Literal[
        "باورهای دینی",
        "اخلاق اسلامی",
        "احکام و فقه",
        "تاریخ اسلام و شخصیت ها",
        "منابع دینی",
        "دین و جامعه/سیاست",
        "عرفان و معنویت",
        "هیچکدام",
    ] = Field(
        ...,
        description="Predicted category label",
        example="اخلاق اسلامی",
    )
