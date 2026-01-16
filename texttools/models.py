from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


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
            raise ValueError(f"Parent category {parent_name} not found")

        node_data = {
            "name": name,
            "description": description if description else "No description provided",
            "level": parent.level + 1,
            "parent": parent,
        }

        new_node = Node(**node_data)
        parent.children[name] = new_node
        self._all_nodes[name] = new_node

    def remove_node(self, name: str, remove_children: bool = True) -> None:
        if name == "root":
            raise ValueError("Cannot remove the root node")

        node = self.get_node(name)
        if not node:
            raise ValueError(f"Category: {name} not found")

        if remove_children:
            # Recursively remove children
            for child_name in list(node.children.keys()):
                self.remove_node(child_name)

        else:
            for child in list(node.children.values()):
                node.parent.children[child.name] = child
                child.parent = node.parent

        del node.parent.children[name]
        del self._all_nodes[name]
