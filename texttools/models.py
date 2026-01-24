from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolOutputMetadata(BaseModel):
    tool_name: str
    processed_at: datetime = Field(default_factory=datetime.now)
    execution_time: float | None = None


class ToolOutput(BaseModel):
    result: Any = None
    analysis: str | None = None
    logprobs: list[dict[str, Any]] | None = None
    errors: list[str] = []
    metadata: ToolOutputMetadata | None = None

    def is_successful(self) -> bool:
        return not self.errors and self.result is not None

    def to_dict(self, exclude_none: bool = False) -> dict:
        return self.model_dump(exclude_none=exclude_none)

    def to_json(self, indent: int = 2, exclude_none: bool = False) -> str:
        return self.model_dump_json(indent=indent, exclude_none=exclude_none)


class Node(BaseModel):
    name: str
    description: str | None
    level: int
    children: dict[str, Node] | None = Field(default_factory=dict)


class CategoryTree:
    def __init__(self):
        self._root = Node(name="root", description="root", level=0)
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
        }

        new_node = Node(**node_data)
        parent.children[name] = new_node
        self._all_nodes[name] = new_node

    def _find_parent(self, name: str) -> Node | None:
        def traverse(node: Node) -> Node | None:
            if name in node.children:
                return node
            for child in node.children.values():
                found = traverse(child)
                if found:
                    return found
            return None

        if name == "root":
            return None

        return traverse(self._root)

    def remove_node(self, name: str, remove_children: bool = True) -> None:
        if name == "root":
            raise ValueError("Cannot remove the root node")

        node = self.get_node(name)
        if not node:
            raise ValueError(f"Category: {name} not found")

        parent = self._find_parent(name)
        if not parent and name != "root":
            raise ValueError("Parent not found, tree inconsistent")

        if remove_children:
            # Recursively remove children
            for child_name in list(node.children.keys()):
                self.remove_node(child_name, remove_children=True)
        else:
            # Move children to parent (grandparent for the children)
            for child_name, child in list(node.children.items()):
                if child_name in parent.children:
                    raise ValueError(f"Name conflict when moving child {child_name}")
                parent.children[child_name] = child

                # Update levels for moved subtree
                def update_levels(n: Node, new_level: int):
                    n.level = new_level
                    for c in n.children.values():
                        update_levels(c, new_level + 1)

                update_levels(child, parent.level + 1)

        del parent.children[name]
        del self._all_nodes[name]

    def dump_tree(self) -> dict:
        return self._root.model_dump()

    def _index_subtree(self, node: Node):
        if node.name in self._all_nodes:
            raise ValueError(f"Duplicate node name: {node.name}")

        self._all_nodes[node.name] = node

        for child in node.children.values():
            self._index_subtree(child)

    @classmethod
    def from_dict(cls, root: dict) -> CategoryTree:
        tree = cls()
        tree._root = Node.model_validate(root)
        tree._all_nodes = {}
        tree._index_subtree(tree._root)
        return tree
