import random
from collections import namedtuple, defaultdict
from os import path
from datetime import datetime
import csv
from contextlib import ContextDecorator
import time

import torch as T

from . import config

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


class MetricsWriter(object):

    def __init__(self):
        stats_filename = datetime.now().strftime('%Y%m%d_%H%M') + ".csv"
        stats_path = path.join(config.DATA_DIR, "stats", stats_filename)
        self._csv_file = open(stats_path, "w")
        self._csv_writer =  None

    def _init_writer(self, field_names):
        writer = csv.DictWriter(self._csv_file, fieldnames=field_names)
        writer.writeheader()
        return writer

    def write(self, metrics):
        if not self._csv_writer:
            self._csv_writer = self._init_writer(metrics.keys())
        self._csv_writer.writerow(metrics)
        self._csv_file.flush()

class Timer(ContextDecorator):
    _TIMES = defaultdict(list)

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, *exc):
        exec_time = (time.time() - self._start_time) * 1000.
        self._TIMES[self._name].append(exec_time)

    @classmethod
    def get_times_and_reset(cls):
        mean_times = {f"t_{name}": sum(times)/len(times) \
                        for name, times in cls._TIMES.items()}
        cls._TIMES = defaultdict(list)
        return mean_times


def product(iterable):
    p = 1
    for n in iterable:
        p *= n
    return p



