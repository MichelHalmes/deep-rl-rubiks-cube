import random
from collections import namedtuple


import torch as T

Transition = namedtuple("Transition",
                        ["state", "action", "next_state", "reward", "done"])

MAX_SIZE = 10000

class ReplayMemory(object):

    def __init__(self, max_size=MAX_SIZE):
        self._max_size = max_size
        self._memory = []
        self._idx = 0

    def push(self, *args):
        if len(self._memory) < self._max_size:
            self._memory.append(None)
        self._memory[self._idx] = Transition(*args)
        self._idx = (self._idx + 1) % self._max_size

    def sample(self, batch_size):
        transitions = random.sample(self._memory, batch_size)
        # We transpose a list of Transitions to get a Transition of lists
        batch = Transition(*(T.stack(items) for items in zip(*transitions)))
        return batch

    def __len__(self):
        return len(self._memory)

def product(iterable):
    p = 1
    for n in iterable:
        p *= n
    return p



