import torch
from torchtext.data import RawField
from torchtext.vocab import Vocab
from collections import Counter
import types


class AnchoredLabelField(RawField):
    def __init__(self):
        super(AnchoredLabelField, self).__init__()
        self.vocab = None

    def process(self, example, device=None):
        example = self.numericalize(example)
        tensor = self.pad(example, device)
        return tensor

    def pad(self, example, device):
        n_labels = len(self.vocab)

        #print(flush=True)
        #print(example, flush=True)

        n_nodes, n_tokens = len(example[1]), example[0]
        #print(n_nodes, n_tokens, n_labels, flush=True)
        #print(flush=True)

        tensor = torch.full([n_nodes, n_tokens, n_labels + 1], 0, dtype=torch.long, device=device)
        for i_node, node in enumerate(example[1]):
            for anchor, rule in node:
                tensor[i_node, anchor, rule + 1] = 1

        return tensor

    def numericalize(self, arr):
        def multi_map(array, function):
            if isinstance(array, tuple):
                return (array[0], function(array[1]))
            elif isinstance(array, list):
                return [multi_map(a, function) for a in array]
            else:
                return array

        if self.vocab is not None:
            arr = multi_map(arr, lambda x: self.vocab.stoi[x] if x in self.vocab.stoi else 0)

        return arr
