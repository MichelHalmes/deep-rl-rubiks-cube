
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from .utils import product
from . import config as cfg

T.manual_seed(0)

def get_transform_f(colors):
    lookup_table = {col: idx for idx, col in enumerate(colors)}
    lookup_f = lambda col: lookup_table[col]
    lookup_fv = np.vectorize(lookup_f)
    nb_colors = len(colors)
    
    def transform_state(sides):
        sides = np.asarray(sides)
        sides = lookup_fv(sides)
        tensor = T.tensor(sides)
        tensor = tensor.permute(1, 2, 0)  # (S,X,Y)=(6,3,3) -> (X,Y,S)=(3,3,6)
        tensor = F.one_hot(tensor, nb_colors).float()
        return tensor

    return transform_state

def swap_cube_axes(tensor):
    # tensor: (B,X,Y,Side,Color) = (B,3,3,6,6)
    B, X, Y, S, C = tensor.shape
    new_size = [B, X*Y, S, C]
    tensor = tensor.view(new_size)  # (B,9,6,6)
    return tensor


def flatten(tensor):
    return tensor.view([tensor.size(0), -1])


class FeedForwardBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, output_size)
        self.ln = nn.LayerNorm(output_size)

    def forward(self, x):
        if self.hidden.in_features == self.hidden.out_features:  # Use residual connections
            return x + F.relu(self.ln(self.hidden(x)))
        else:
            return F.selu(self.ln(self.hidden(x)))


class DQN(nn.Module):

    def __init__(self, input_shape, output_size):
        super().__init__()
        assert input_shape == [3, 3, 6, 6]
        # 1x1 convolutions over 6x6x9 "image", the channels represent thus the 9 kubies per side
        # This introduces invariance over which side and in which color a certain pattern appears
        self._conv = nn.Conv2d(in_channels=3*3, out_channels=cfg.CONV_NB_KERNELS, kernel_size=1)
        in_width = 6*6*cfg.CONV_NB_KERNELS  # Size after Flatten
        self._ffwd_layers = self._init_ffwd_layers(in_width)
        self._head = nn.Linear(cfg.LAYER_SIZES[-1], output_size)

    def _init_ffwd_layers(self, in_width):
        ffwd_layers = []
        for out_width in cfg.LAYER_SIZES:
            ffwd_layers.append(
                FeedForwardBlock(in_width, out_width))
            in_width = out_width
        return nn.Sequential(*ffwd_layers)

    def forward(self, x):
        assert len(x.shape) == 5 and x.shape[1:] == T.Size([3, 3, 6, 6]), \
                    f"Expected (B,N,N,S,C), got: {x.shape}"
        y = swap_cube_axes(x)
        y = F.selu(self._conv(y))
        y = flatten(y)
        y = self._ffwd_layers(y)

        return F.softplus(self._head(y))  # All rewards and state actions must be >= 0