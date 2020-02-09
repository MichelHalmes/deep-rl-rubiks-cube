
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from .utils import product
from . import config

def get_transform_f(colors):
    lookup_table = {col: idx for idx, col in enumerate(colors)}
    lookup_f = lambda col: lookup_table[col]
    lookup_fv = np.vectorize(lookup_f)
    nb_colors = len(colors)
    
    def transform_state(sides):
        sides = np.asarray(sides)
        sides = lookup_fv(sides)
        tensor = T.tensor(sides)
        tensor = F.one_hot(tensor, nb_colors).float()
        return tensor

    return transform_state


def flatten(tensor):
    return tensor.view([tensor.size(0), -1])


class FeedForwardBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, output_size)
        self.ln = nn.LayerNorm(output_size)

    def forward(self, x):
        return F.selu(self.ln(self.hidden(x)))



class DQN(nn.Module):

    def __init__(self, input_shape, output_size):
        super().__init__()
        ffwd_layers = []
        in_width = product(input_shape)  # Size after Flatten
        for out_width in config.LAYER_SIZES:
            ffwd_layers.append(
                FeedForwardBlock(in_width, out_width))
            in_width = out_width
        self._ffwd_layers = nn.Sequential(*ffwd_layers)

        self._head = nn.Linear(in_width, output_size)

    def forward(self, x):
        assert len(x.shape) == 5, f"Expected (B,N,N,C,C), got: {x.shape}"
        y = flatten(x)
        for layer in self._ffwd_layers:
            y = layer(y)

        return F.softplus(self._head(y))  # All rewards and state actions must be >= 0