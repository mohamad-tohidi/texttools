from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .core import TokenUsage


class ToolOutputMetadata(BaseModel):
    tool_name: str
    processed_by: str | None = None
    processed_at: datetime = Field(default_factory=datetime.now)
    execution_time: float | None = None
    token_usage: TokenUsage | None = None


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


class CategoryNode(BaseModel):
    name: str
    description: str | None
    depth: int
    children: dict[str, CategoryNode] | None = Field(default_factory=dict)


class CategoryTree:
    def __init__(self):
        self._root = CategoryNode(name="root", description="root", depth=0)
        self._all_nodes = {"root": self._root}

    def get_all_nodes(self) -> dict[str, CategoryNode]:
        return self._all_nodes

    def get_max_depth(self) -> int:
        return max(node.depth for node in self._all_nodes.values())

    def get_node(self, name: str) -> CategoryNode | None:
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
            "depth": parent.depth + 1,
        }

        new_node = CategoryNode(**node_data)
        parent.children[name] = new_node
        self._all_nodes[name] = new_node

    def _find_parent(self, name: str) -> CategoryNode | None:
        def traverse(node: CategoryNode) -> CategoryNode | None:
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

                # Update depths for moved subtree
                def update_depths(n: CategoryNode, new_depth: int):
                    n.depth = new_depth
                    for c in n.children.values():
                        update_depths(c, new_depth + 1)

                update_depths(child, parent.depth + 1)

        del parent.children[name]
        del self._all_nodes[name]

    def dump_tree(self) -> dict:
        return self._root.model_dump()

    def _index_subtree(self, node: CategoryNode):
        if node.name in self._all_nodes:
            raise ValueError(f"Duplicate node name: {node.name}")

        self._all_nodes[node.name] = node

        for child in node.children.values():
            self._index_subtree(child)

    @classmethod
    def from_dict(cls, root: dict) -> CategoryTree:
        tree = cls()
        tree._root = CategoryNode.model_validate(root)
        tree._all_nodes = {}
        tree._index_subtree(tree._root)
        return tree
