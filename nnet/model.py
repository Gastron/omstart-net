import torch.nn as nn
import numpy as np

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers = 1,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)
        self.rnn = nn.LSTM(input_size = embedding_size, hidden_size = hidden_size,
                num_layers = num_layers)
        self.project_back_to_vocab = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 #Recommended somewhere
        self.embedder.weight.data.uniform_(-initrange, initrange)
        self.project_back_to_vocab.bias.data.zero_()
        self.project_back_to_vocab.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, seq_lengths=None, hidden_state=None):
        batch_size, seq_len = inputs.size()
        if seq_lengths is None:
            #assume all last till end
            seq_lengths = torch.ones((batch_size,1), dtype=torch.long)*seq_len
        if hidden_state is None:
            hidden_state = self.get_hidden_init(batch_size)
        embeddings = self.embedder(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, seq_lengths, batch_first=True)
        rnn_output, hidden_output = self.rnn(packed, hidden_state)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
         
        projected = self.project_back_to_vocab(unpacked)
        return projected, hidden_output

    def get_hidden_init(self, batch_size): #batch_size=batch size
        weight = next(self.parameters())
        return (weight.new_zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size),
                weight.new_zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))

