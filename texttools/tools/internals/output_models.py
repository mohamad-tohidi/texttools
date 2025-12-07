import logging
from typing import Type, Any, Literal

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger("texttools.models")


class ToolOutput(BaseModel):
    result: Any = None
    analysis: str = ""
    logprobs: list[dict[str, Any]] = []
    errors: list[str] = []

    def __repr__(self) -> str:
        return f"ToolOutput(result_type='{type(self.result)}', result='{self.result}', analysis='{self.analysis}', logprobs='{self.logprobs}', errors='{self.errors}'"


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


class Node(BaseModel):
    node_id: int
    name: str
    level: int
    parent_id: int | None
    description: str | None = None


class CategoryTree:
    def __init__(self, tree_name):
        self.root = Node(node_id=0, name=tree_name, level=0, parent_id=None)
        self.node_list: list[Node] = [self.root]
        self.new_id = 1

    def add_category(self, category_name, parent_name: str | None = None) -> None:
        if self.find_category(category_name):
            logger.error(
                f"{category_name} has been choosed for another category before"
            )
            return
        if parent_name:
            parent_node = self.find_category(parent_name)
            if not parent_node:
                logger.error(f"Parent category '{parent_name}' not found")
                return
            parent_id = parent_node.node_id
            level = parent_node.level + 1
        else:
            level = 1
            parent_id = 0
        self.node_list.append(
            Node(
                node_id=self.new_id,
                name=category_name,
                level=level,
                parent_id=parent_id,
            )
        )
        self.new_id += 1
        logger.info(
            Node(
                node_id=self.new_id,
                name=category_name,
                level=level,
                parent_id=parent_id,
            )
        )

    def add_description(self, category, description):
        if isinstance(category, str):
            node = self.find_category(category)
        elif isinstance(category, int):
            node = self.find_category_by_id(category)
        try:
            node.description = description
        except NameError:
            logger.error(f"There is no category with this id/name: {category}")

    def find_all(self) -> list[Node]:
        return self.node_list

    def find_category(self, node_name: str) -> Node | None:
        for node in self.node_list:
            if node_name == node.name:
                return node
        return None

    def find_category_by_id(self, node_id: int) -> Node | None:
        for node in self.node_list:
            if node_id == node.node_id:
                return node
        return None

    def find_categories_by_parent_id(self, parent_id: int) -> list[Node] | None:
        nodes = []
        for node in self.node_list:
            if parent_id == node.parent_id:
                nodes.append(node)
        if nodes:
            return nodes
        return None

    def remove_category(self, node_id: str) -> None:
        child_node = self.find_category(node_id)
        if child_node:
            self.node_list.remove(child_node)
        else:
            logger.error(f"Parent node with value '{node_id}' not found.")

    def dump_tree(self) -> dict:
        def build_dict(node: Node) -> dict:
            children = [
                build_dict(child)
                for child in self.node_list
                if child.parent_id == node.node_id
            ]
            return {
                "node_id": node.node_id,
                "name": node.name,
                "level": node.level,
                "parent_id": node.parent_id,
                "children": children,
            }

        return {"category_tree": build_dict(self.root)["children"]}

    def level_count(self) -> int:
        return max([item.level for item in self.node_list])


class Entity(BaseModel):
    text: str = Field(description="The exact text of the entity")
    type: str = Field(description="The type of the entity")


class EntityDetectorOutput(BaseModel):
    result: list[Entity] = Field(description="List of all extracted entities")
