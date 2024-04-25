import torch
import numpy as np

from Tokenizer import get_device

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):

        positional_encoding = np.zeros((self.max_sequence_length, self.d_model))

        for pos in range(self.max_sequence_length):
            for i in range(0, self.d_model, 2):
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.d_model)))

                if i + 1 < self.d_model:
                    positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / self.d_model)))

        return torch.from_numpy(positional_encoding).float().to(get_device())

    def forward(self, x):
        return x + self.positional_encoding[:x.size(1), :]
