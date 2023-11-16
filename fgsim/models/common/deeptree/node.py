from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class Node:
    idxs: torch.Tensor
    parent: Optional[Node] = None
    children: List[Node] = field(default_factory=lambda: [])

    def __post_init__(self):
        assert torch.long == self.idxs.dtype
        if self.children is None:
            self.children = []

    def add_child(self, child: Node):
        assert child.parent is None
        self.children.append(child)
        child.parent = self

    def get_ancestors(self) -> List[Node]:
        if self.parent is None:
            return []
        else:
            return [self.parent] + self.parent.get_ancestors()

    def recur_descendants(self):
        return self.children + [
            coc for child in self.children for coc in child.recur_descendants()
        ]

    def get_root(self) -> Node:
        cur = self
        while cur.parent is not None:
            cur = cur.parent
        return cur

    def get_node_list(self) -> List[Node]:
        root = self.get_root()
        node_list = [root] + root.recur_descendants()
        return node_list

    def __hash__(self):
        return hash(
            tuple([self.idxs, self.parent] + [e.idxs for e in self.children])
        )

    def __repr__(self) -> str:
        return f"Node({self.idxs[0]}->{self.idxs[-1]})"
