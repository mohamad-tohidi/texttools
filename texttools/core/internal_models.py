from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AnalyzeUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TokenUsage(BaseModel):
    completion_usage: CompletionUsage = CompletionUsage()
    analyze_usage: AnalyzeUsage = AnalyzeUsage()
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        new_completion_usage = CompletionUsage(
            prompt_tokens=self.completion_usage.prompt_tokens
            + other.completion_usage.prompt_tokens,
            completion_tokens=self.completion_usage.completion_tokens
            + other.completion_usage.completion_tokens,
            total_tokens=self.completion_usage.total_tokens
            + other.completion_usage.total_tokens,
        )
        new_analyze_usage = AnalyzeUsage(
            prompt_tokens=self.analyze_usage.prompt_tokens
            + other.analyze_usage.prompt_tokens,
            completion_tokens=self.analyze_usage.completion_tokens
            + other.analyze_usage.completion_tokens,
            total_tokens=self.analyze_usage.total_tokens
            + other.analyze_usage.total_tokens,
        )
        total_tokens = (
            new_completion_usage.total_tokens + new_analyze_usage.total_tokens
        )

        return TokenUsage(
            completion_usage=new_completion_usage,
            analyze_usage=new_analyze_usage,
            total_tokens=total_tokens,
        )


class OperatorOutput(BaseModel):
    result: Any
    analysis: str | None
    logprobs: list[dict[str, Any]] | None
    processed_by: str
    token_usage: TokenUsage | None = None


class Str(BaseModel):
    result: str = Field(
        ..., description="The output string", json_schema_extra={"example": "text"}
    )


class Bool(BaseModel):
    result: bool = Field(
        ...,
        description="Boolean indicating the output state",
        json_schema_extra={"example": True},
    )


class ListStr(BaseModel):
    result: list[str] = Field(
        ...,
        description="The output list of strings",
        json_schema_extra={"example": ["text_1", "text_2", "text_3"]},
    )


class ListDictStrStr(BaseModel):
    result: list[dict[str, str]] = Field(
        ...,
        description="List of dictionaries containing string key-value pairs",
        json_schema_extra={
            "example": [
                {"text": "Mohammad", "type": "PER"},
                {"text": "Iran", "type": "LOC"},
            ]
        },
    )


class ReasonListStr(BaseModel):
    reason: str = Field(..., description="Thinking process that led to the output")
    result: list[str] = Field(
        ...,
        description="The output list of strings",
        json_schema_extra={"example": ["text_1", "text_2", "text_3"]},
    )


# Create CategorizerOutput with dynamic categories
def create_dynamic_model(allowed_values: list[str]) -> type[BaseModel]:
    literal_type = Literal[*allowed_values]

    CategorizerOutput = create_model(
        "CategorizerOutput",
        reason=(
            str,
            Field(
                ..., description="Explanation of why the input belongs to the category"
            ),
        ),
        result=(literal_type, Field(..., description="Predicted category label")),
    )

    return CategorizerOutput
