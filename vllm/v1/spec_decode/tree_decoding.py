from typing import List, Dict, Any, Optional
from collections import deque

import torch


class SequenceTree:

    def __init__(self):
        self.root: Dict[int, Dict[str, Any]] = {}
        self._flattened_sequence: Optional[List[int]] = None
        self._seqs: List[List[int]] = []

    def add_sequence(self, sequence: List[int]):
        self._seqs.append(sequence)

        current_level_nodes = self.root
        for number in sequence:
            if number not in current_level_nodes:
                current_level_nodes[number] = {'children': {}, 'index': None}
            current_level_nodes = current_level_nodes[number]['children']

    def get_sequences(self) -> List[List[int]]:
        return self._seqs

    def __get_indices(self, sequence: List[int]) -> List[int]:
        current_level_nodes = self.root
        indices = []
        for number in sequence:
            if number not in current_level_nodes:
                return indices
            indices.append(current_level_nodes[number]['index'])
            current_level_nodes = current_level_nodes[number]['children']
        return indices

    def verify(self, sequence: List[int]) -> List[int]:
        assert self._flattened_sequence is not None, "Please call flat() before calling verify()"
        assert len(sequence) == len(
            self._flattened_sequence
        ), "The sequence to verify must have the same length as the flattened sequence"
        assert len(
            self.root.keys()) == 1, "There must be exactly one root node"

        current_level_nodes = self.root[list(self.root.keys())[0]]
        seq_idx = 0
        indices = [seq_idx]
        while sequence[seq_idx] in current_level_nodes['children']:
            current_level_nodes = current_level_nodes['children'][
                sequence[seq_idx]]
            seq_idx = current_level_nodes['index']
            indices.append(seq_idx)

        return indices

    def __create_tree_mask(self) -> torch.Tensor:
        assert self._flattened_sequence is not None, "Please call flat() before calling _create_tree_mask()"

        max_sequence_length = len(self._flattened_sequence)
        mask = torch.eye(max_sequence_length, dtype=torch.bool)
        mask[:, 0] = True

        paths = [self.__get_indices(seq) for seq in self._seqs]
        for path in paths:
            print("path", path)
            for i in range(1, len(path)):
                for j in range(i + 1, len(path)):
                    mask[path[j], path[i]] = True

        return mask

    def flat(self) -> List[int]:
        self._flattened_sequence = []
        queue = deque()
        current_index = 0

        for number in self.root.keys():
            node_data = self.root[number]
            queue.append((number, node_data))

        while queue:
            number, node_data = queue.popleft()
            node_data['index'] = current_index
            self._flattened_sequence.append(number)
            current_index += 1

            children_dict = node_data['children']
            for child_number in children_dict.keys():
                child_node_data = children_dict[child_number]
                queue.append((child_number, child_node_data))

        return self._flattened_sequence, self.__create_tree_mask()

    def __str__(self) -> str:
        import json

        def default_serializer(obj):
            return str(obj)

        try:
            return json.dumps(self.root, indent=2, default=default_serializer)
        except TypeError as e:
            return f"Could not serialize tree structure: {e}\nRoot dictionary: {self.root}"


# if __name__ == "__main__":
#     tree = SequenceTree()

#     tree.add_sequence([16, 13, 578, 7301])
#     tree.add_sequence([16, 13, 578, 374])
#     tree.add_sequence([16, 13, 3639])
#     tree.add_sequence([16, 320, 791])

#     print(tree)

#     flattened_sequence, mask = tree.flat()
#     print(f"\nFlattened Sequence: {flattened_sequence}")
#     print(f"Mask:")
#     print(mask.to(torch.int))

#     scorer_ids = [13, 578, 321, 374, 28, 19, 250, 55]
#     res = tree.verify(scorer_ids)
#     print(f"\nScorer IDs: {scorer_ids}")
#     print(f"Verified Indices: {res}")
