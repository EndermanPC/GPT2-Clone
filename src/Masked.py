import numpy as np
import torch

class MaskedSelfAttention(torch.nn.Module):
    def __init__(self, embedding_dimension, head_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.query_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.key_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.value_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        attention_weights = torch.matmul(query, key.transpose(-2, -1))

        attention_weights = attention_weights / np.sqrt(self.head_dimension)

        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        attention_scores = self.softmax(attention_weights)

        return torch.bmm(attention_scores, value)

class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)])

        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

    def forward(self, x, mask):
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        return self.output_layer(concatenated_self_attention_outputs)
