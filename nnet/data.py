import os
from io import open
import torch
import torch.utils.data
import random
import numpy as np


class InMemoryTextDataSet(torch.utils.data.Dataset):
    # Sometimes you can fit all data in memory.
    def __init__(self, path):
        self.inputs = []
        self.targets = []
        self.lengths = []
        self.padding_token = 0
        different_tokens = set()
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                tokens = [int(token) for token in line.strip().split()]
                assert self.padding_token not in tokens
                different_tokens.update(tokens)
                self.inputs.append(tokens[:-1])
                self.targets.append(tokens[1:])
                self.lengths.append(len(tokens)-1)
        self.vocab_size = len(different_tokens) +1#+1 for padding_token

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.lengths[index]
    
    def __len__(self):
        return len(self.inputs)

def sort_batch(inputs, targets, lengths):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    inputs_tensor = inputs[perm_idx]
    targets_tensor = targets[perm_idx]
    return inputs_tensor, targets_tensor, seq_lengths

def pad_and_sort_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (inputs, targets, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest, 
    """
    batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    inputs, targets, lengths = batch_split[0], batch_split[1], batch_split[2]
    max_length = max(lengths)

    padded_inputs = np.zeros((batch_size, max_length), dtype=np.long)
    padded_targets = np.zeros((batch_size, max_length), dtype=np.long)
    for i, l in enumerate(lengths):
        padded_inputs[i, 0:l] = inputs[i][0:l]
        padded_targets[i, 0:l] = targets[i][0:l]

    return sort_batch(torch.LongTensor(padded_inputs), 
            torch.LongTensor(padded_targets), 
            torch.LongTensor(lengths))
