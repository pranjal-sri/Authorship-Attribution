import json
from pathlib import Path
from typing import Optional, Union, Dict
from collections import defaultdict
from .tree import Tree

class TreeSerializer:
    """Handles serialization and deserialization of Tree structures"""
    @staticmethod
    def tree_to_dict(node: 'Tree') -> Optional[Dict]:
        if node is None:
            return None
        return {
            'start': node.start,
            'end': node.end,
            'text': node.text,
            'left': TreeSerializer.tree_to_dict(node.left),
            'right': TreeSerializer.tree_to_dict(node.right)
        }

    @staticmethod
    def dict_to_tree(tree_dict: Optional[Dict], with_height = True) -> Optional['Tree']:
        height_map = defaultdict(list)

        def _dict_to_tree(data: Optional[Dict]) -> Optional['Tree']:
            if data is None:
                return None, -1

            node = Tree()
            node.start = data['start']
            node.end = data['end']
            node.text = data['text']

            left_child, height_l = _dict_to_tree(data['left'])
            right_child, height_r = _dict_to_tree(data['right'])
            height = max(height_l, height_r)+1

            if with_height:
                height_map[height].append(node)

            if left_child:
                node.left = left_child
                left_child.parent = node
            if right_child:
                node.right = right_child
                right_child.parent = node

            return node, height

        tree, tree_height = _dict_to_tree(tree_dict)
        if with_height:
            return tree, height_map, tree_height
        else:
            return tree

    @staticmethod
    def save_json(tree: 'Tree', filepath: Union[str, Path]) -> None:
        """Save tree to a JSON file"""
        filepath = Path(filepath)
        tree_dict = TreeSerializer.tree_to_dict(tree)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, indent=2)

    @staticmethod
    def load_json_tree(filepath: Union[str, Path], with_height = True) -> 'Tree':
        """Load tree from a JSON file"""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            tree_dict = json.load(f)
        return TreeSerializer.dict_to_tree(tree_dict, with_height)