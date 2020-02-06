
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from utils import product



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

class DQN(nn.Module):

    def __init__(self, input_shape, output_size):
        super().__init__()
        flatten_size = product(input_shape)
        self.hidden_1 = nn.Linear(flatten_size, 512)
        self.ln_1 = nn.LayerNorm(512)
        self.hidden_2 = nn.Linear(512, 256)
        self.ln_2 = nn.LayerNorm(256)
        self.hidden_3 = nn.Linear(256, 128)
        self.ln_3 = nn.LayerNorm(128)
        self.head = nn.Linear(128, output_size)

    def forward(self, x):
        assert len(x.shape) == 5, f"Expected (B,N,N,C,C), got: {x.shape}"
        x = x.view([x.size(0), -1])  # Flatten
        x = F.relu(self.ln_1(self.hidden_1(x)))
        x = F.relu(self.ln_2(self.hidden_2(x)))
        x = F.relu(self.ln_3(self.hidden_3(x)))

        return F.softplus(self.head(x))  # All rewards and state actions must be >= 0