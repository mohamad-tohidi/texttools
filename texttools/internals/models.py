from datetime import datetime
from typing import Type, Any, Literal

from pydantic import BaseModel, Field, create_model


class ToolOutput(BaseModel):
    result: Any = None
    logprobs: list[dict[str, Any]] = []
    analysis: str = ""
    process: str | None = None
    processed_at: datetime = datetime.now()
    execution_time: float | None = None
    errors: list[str] = []

    def __repr__(self) -> str:
        return f"""
        ToolOutput(process='{self.process}', result_type='{type(self.result)}', 
        result='{self.result}', analysis='{self.analysis}', 
        logprobs='{self.logprobs}', errors='{self.errors}', 
        processed_at='{self.processed_at}', execution_time='{self.execution_time}'
        """


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


class Node(BaseModel):
    node_id: int
    name: str
    level: int
    parent_id: int | None
    description: str


class CategoryTree:
    def __init__(self, tree_name):
        self._root = Node(
            node_id=0, name=tree_name, level=0, parent_id=None, description="Root node"
        )
        self._all_nodes: list[Node] = [self._root]
        self._new_id = 1

    def get_all_nodes(self) -> list[Node]:
        return self._all_nodes

    def get_level_count(self) -> int:
        return max([item.level for item in self._all_nodes])

    def get_node(self, identifier: int | str) -> Node | None:
        if isinstance(identifier, str):
            for node in self.get_all_nodes():
                if node.name == identifier:
                    return node
            return None
        elif isinstance(identifier, int):
            for node in self.get_all_nodes():
                if node.node_id == identifier:
                    return node
            return None
        else:
            return None

    def get_children(self, parent_node: Node) -> list[Node] | None:
        children = [
            node
            for node in self.get_all_nodes()
            if parent_node.node_id == node.parent_id
        ]
        return children if children else None

    def add_node(
        self,
        node_name: str,
        parent_name: str | None = None,
        description: str | None = None,
    ) -> None:
        if self.get_node(node_name):
            raise ValueError(f"{node_name} has been chosen for another category before")

        if parent_name:
            parent_node = self.get_node(parent_name)
            if not parent_node:
                raise ValueError(f"Parent category '{parent_name}' not found")
            parent_id = parent_node.node_id
            level = parent_node.level + 1
        else:
            level = 1
            parent_id = 0

        node_data = {
            "node_id": self._new_id,
            "name": node_name,
            "level": level,
            "parent_id": parent_id,
            "description": description if description else "No description provided",
        }

        self._all_nodes.append(Node(**node_data))
        self._new_id += 1

    def remove_node(self, identifier: int | str) -> None:
        node = self.get_node(identifier)

        if node:
            # Remove node's children recursively
            children = self.get_children(node)

            if not children:
                self._all_nodes.remove(node)
                return

            for child in children:
                self.remove_node(child.name)

            self._all_nodes.remove(node)
        else:
            raise ValueError(f"Node with identifier: '{identifier}' not found.")

    def dump_tree(self) -> dict:
        def build_dict(node: Node) -> dict:
            children = [
                build_dict(child)
                for child in self._all_nodes
                if child.parent_id == node.node_id
            ]
            return {
                "node_id": node.node_id,
                "name": node.name,
                "level": node.level,
                "parent_id": node.parent_id,
                "children": children,
            }

        return {"category_tree": build_dict(self._root)["children"]}


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
