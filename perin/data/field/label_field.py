import torch
import torchtext
from torchtext.data import RawField
from collections import Counter
from torchtext.vocab import Vocab


class LabelField(RawField):
    def __self__(self, preprocessing):
        super(LabelField, self).__init__(preprocessing=preprocessing)
        self.vocab = None

    def build_vocab(self, *args, **kwargs):
        sources = []
        for arg in args:
            if isinstance(arg, torch.utils.data.Dataset):
                sources += [arg.get_examples(name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        counter = Counter()
        for data in sources:
            for x in data:
                counter.update(x)

        self.vocab = Vocab(counter, specials=[])
        print(self.vocab.freqs)

    def process(self, example, device=None):
        tensor, lengths = self.numericalize(example, device=device)
        return tensor, lengths

    def numericalize(self, example, device=None):
        example = [self.vocab.stoi[x] + 1 for x in example]
        length = torch.LongTensor([len(example)], device=device).squeeze(0)
        tensor = torch.LongTensor(example, device=device)

        return tensor, length
