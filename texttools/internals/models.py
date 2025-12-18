from __future__ import annotations

from datetime import datetime
from typing import Type, Literal, Any

from pydantic import BaseModel, Field, create_model


class ToolOutputMetadata(BaseModel):
    tool_name: str
    processed_at: datetime = datetime.now()
    execution_time: float | None = None


class ToolOutput(BaseModel):
    result: Any = None
    analysis: str | None = None
    logprobs: list[dict[str, Any]] | None = None
    errors: list[str] = []
    metadata: ToolOutputMetadata | None = None

    def __repr__(self) -> str:
        return f"ToolOutput({self.model_dump_json(indent=2)})"


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
        ..., description="The output list of strings", example=["text_1", "text_2"]
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
        ..., description="The output list of strings", example=["text_1", "text_2"]
    )


class Node:
    def __init__(self, name: str, description: str, level: int, parent: Node | None):
        self.name = name
        self.description = description
        self.level = level
        self.parent = parent
        self.children = {}


class CategoryTree:
    def __init__(self):
        self._root = Node(name="root", description="root", level=0, parent=None)
        self._all_nodes = {"root": self._root}

    def get_all_nodes(self) -> dict[str, Node]:
        return self._all_nodes

    def get_level_count(self) -> int:
        return max(node.level for node in self._all_nodes.values())

    def get_node(self, name: str) -> Node | None:
        return self._all_nodes.get(name)

    def add_node(
        self,
        name: str,
        parent_name: str,
        description: str | None = None,
    ) -> None:
        if self.get_node(name):
            raise ValueError(f"Cannot add {name} category twice")

        parent = self.get_node(parent_name)

        if not parent:
            raise ValueError(f"Parent category '{parent_name}' not found")

        node_data = {
            "name": name,
            "description": description if description else "No description provided",
            "level": parent.level + 1,
            "parent": parent,
        }

        new_node = Node(**node_data)
        parent.children[name] = new_node
        self._all_nodes[name] = new_node

    def remove_node(self, name: str) -> None:
        if name == "root":
            raise ValueError("Cannot remove the root node")

        node = self.get_node(name)
        if not node:
            raise ValueError(f"Category: '{name}' not found")

        for child_name in list(node.children.keys()):
            self.remove_node(child_name)

        if node.parent:
            del node.parent.children[name]

        del self._all_nodes[name]


# This function is needed to create CategorizerOutput with dynamic categories
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
