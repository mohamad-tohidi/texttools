from typing import Any, Literal, Type

from pydantic import BaseModel, Field, create_model


class OperatorOutput(BaseModel):
    result: Any
    analysis: str | None
    logprobs: list[dict[str, Any]] | None


class Str(BaseModel):
    result: str = Field(..., description="The output string", example="text")


class Bool(BaseModel):
    result: bool = Field(
        ..., description="Boolean indicating the output state", example=True
    )


class ListStr(BaseModel):
    result: list[str] = Field(
        ...,
        description="The output list of strings",
        example=["text_1", "text_2", "text_3"],
    )


class ListDictStrStr(BaseModel):
    result: list[dict[str, str]] = Field(
        ...,
        description="List of dictionaries containing string key-value pairs",
        example=[{"text": "Mohammad", "type": "PER"}, {"text": "Iran", "type": "LOC"}],
    )


class ReasonListStr(BaseModel):
    reason: str = Field(..., description="Thinking process that led to the output")
    result: list[str] = Field(
        ...,
        description="The output list of strings",
        example=["text_1", "text_2", "text_3"],
    )


# Create CategorizerOutput with dynamic categories
def create_dynamic_model(allowed_values: list[str]) -> Type[BaseModel]:
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
