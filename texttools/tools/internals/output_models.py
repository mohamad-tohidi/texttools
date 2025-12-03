from typing import Dict, Literal, Any, List, Optional

from pydantic import BaseModel, Field


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


class Entity(BaseModel):
    text: str = Field(description="The exact text of the entity")
    type: str = Field(description="The type of the entity")


class EntityDetectorOutput(BaseModel):
    result: List[Entity] = Field(description="List of all extracted entities")


class Node(BaseModel):
    id: int
    name: str
    level: int
    parent_id: Optional[int]


class CategoryTree:
    def __init__(self, tree_name):
        self.root = Node(id=0, name=tree_name, level=0, parent_id=None)
        # self.node_list: List[str] = [self.root.name]
        self.node_list: List[Node] = [self.root]
        self.new_id = 1

    def add_category(self, category_name, parent_name: Optional[str] = None) -> None:
        if self.find_category(category_name):
            raise ValueError(
                f"this '{category_name}' has been choosed for another category before"
            )
        if parent_name:
            if self.find_category(parent_name):
                parent_id = self.find_category(parent_name).id
                level = self.find_category(parent_name).level + 1
        else:
            level = 1
            parent_id = 0
        self.node_list.append(
            Node(id=self.new_id, name=category_name, level=level, parent_id=parent_id)
        )
        self.new_id += self.new_id + 1
        print(
            Node(id=self.new_id, name=category_name, level=level, parent_id=parent_id)
        )

    def find_category(self, node_name):
        # node_names_list = [node.name for node in self.node_list]
        for node in self.node_list:
            if node_name == node.name:
                return node
        return False

    def find_category_by_id(self, node_id):
        # node_names_list = [node.name for node in self.node_list]
        for node in self.node_list:
            if node_id == node.id:
                return node
        return False

    def remove_category(self, node_id: str) -> None:
        """
        Remove a child node from a parent
        """
        child_node = self.find_category(node_id)
        if child_node:
            self.node_list.remove(child_node)
        else:
            raise ValueError(f"Parent node with value '{node_id}' not found.")

    def show_tree(self):
        def build_dict(node: Node) -> Dict:
            children = [
                build_dict(child)
                for child in self.node_list
                if child.parent_id == node.id
            ]
            return {
                "id": node.id,
                "name": node.name,
                "level": node.level,
                "parent_id": node.parent_id,
                "children": children,
            }

        return build_dict(self.root)
