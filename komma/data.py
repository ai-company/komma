from math import floor
import torch
import numpy as np

from typing import List

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from alive_progress import alive_bar


PUNCT_BIBLE = {
    "O": 0,
    "COMMA": 1,
    "PERIOD": 0,
    "QUESTION": 0,
}


def encode_text(data: List[str], tokenizer: BertTokenizer):
    inputs, targets = [], []

    with alive_bar(floor(len(data) / 100), title=f"Processing") as bar:
        for l, line in enumerate(data):
            if l % 100 == 0:
                bar()
            # TODO: Data will look like this
            word, label = line.split("\t")
            label = label.strip()

            tokens = tokenizer.tokenize(word)

            x = tokenizer.convert_tokens_to_ids(tokens)

            # > Can you explain this gab in your input data?
            # ... Can you explain THIS??
            y = (len(x) - 1) * [PUNCT_BIBLE["O"]] + [PUNCT_BIBLE[label]]  # !$@#*^!!

            assert len(x) == len(y)

            inputs += x
            targets += y

    return inputs, targets


def create_segments(words, segment_size):
    segments = []

    segment_pad = (
        words[-((segment_size - 1) // 2 - 1) :] + words + words[: segment_size // 2]
    )

    with alive_bar(
        floor((len(segment_pad) - segment_size + 2) / 100), title=f"Processing"
    ) as bar:
        for i in range(len(segment_pad) - segment_size + 2):
            if i % 100 == 0:
                bar()
            segment_new = segment_pad[i : i + segment_size - 1]
            segment_new.insert((segment_size - 1) // 2, 0)

            segments.append(segment_new)

    return np.asarray(segments)


def create_loader(
    text_path, tokenizer, segment_size, batch_size, shuffle=True
) -> DataLoader:
    data = open(text_path, "r", encoding="utf-8").readlines()

    print("Encoding text")
    inputs, targets = encode_text(data, tokenizer)

    assert len(inputs) == len(targets)

    print("Creating segments")
    inputs = create_segments(inputs, segment_size)

    dataset = TensorDataset(
        torch.from_numpy(inputs), torch.from_numpy(np.array(targets).astype(np.float32))
    )

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
