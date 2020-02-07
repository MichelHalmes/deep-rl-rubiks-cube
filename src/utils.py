import random
from collections import namedtuple, defaultdict
from os import path
from datetime import datetime
import csv
from contextlib import ContextDecorator
import time
import math

import torch as T

from . import config

Transition = namedtuple("Transition",
                        ["state", "action", "next_state", "reward", "done"])


class ReplayMemory(object):

    def __init__(self, max_size=config.MEMORY_MAX_SIZE):
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


def fibonacci(n):
    if n <= 1:
        return n
    prev, cur = 0, 1
    for _ in range(1, n):
        prev, cur = cur, prev + cur
    return cur

TrainCfg = namedtuple("TrainCfg", ["episode", "difficulty", "epsilon", "max_steps"])

class TrainSchedule(object):

    def __init__(self):
        self._episode = 0
        self._difficulty = 1
        self._diff_curr_steps = 0
        self._diff_max_steps = config.DIFFICULTY_STEPS

    def iter_train_configs(self):
        while True:
            yield self._get_next_step()

    def _next_episode(self):
        self._episode += 1
        return self._episode

    def _next_difficulty(self):
        # At every difficulty, we remain DIFFICULTY_STEPS longer in the level
        self._diff_curr_steps += 1
        if self._diff_curr_steps == self._diff_max_steps:
            self._difficulty += 1
            self._diff_curr_steps = 0
            self._diff_max_steps = config.DIFFICULTY_STEPS*self._difficulty

        return self._difficulty

    def _next_epsilon(self):
        decay = 1. - (self._diff_curr_steps / self._diff_max_steps)
        epsilon = config.EPS_END + (config.EPS_START - config.EPS_END) * decay**.7
        return round(epsilon, 3)

    def _next_max_steps(self):
        return config.MAX_STEPS * int(math.sqrt(self._difficulty))

    def _get_next_step(self):
        episode = self._next_episode()
        difficulty = self._next_difficulty()
        epsilon = self._next_epsilon()
        max_steps = self._next_max_steps()
        return TrainCfg(episode, difficulty, epsilon, max_steps)


class MetricsWriter(object):
    _MA_NAMES = ["duration", "done"]

    def __init__(self, suffix):
        stats_filename = datetime.now().strftime('%Y%m%d_%H%M') + f"_{suffix}.csv"
        stats_path = path.join(config.DATA_DIR, "stats", stats_filename)
        self._csv_file = open(stats_path, "w")
        self._csv_writer =  None
        self._ma_metrics = None

    def write(self, metrics):
        if not self._csv_writer:  # First call
            self._ma_metrics = self._init_ma_metrics(metrics)
            self._csv_writer = self._init_writer(metrics.keys())

        self._update_timer_metrics(metrics)
        self._update_ma_metrics(metrics)
        self._csv_writer.writerow(metrics)
        self._csv_file.flush()
        self._print_metrics(metrics)

    def _init_writer(self, field_names):
        field_names = list(field_names) + list(self._ma_metrics.keys()) + list(Timer.get_times_and_reset().keys())
        writer = csv.DictWriter(self._csv_file, fieldnames=field_names)
        writer.writeheader()
        return writer

    def _init_ma_metrics(self, metrics):
        ma_metrics = {f"{name}_ma": metrics[name]
                        for name in self._MA_NAMES}
        return ma_metrics

    def _update_ma_metrics(self, metrics):
        for name in self._MA_NAMES:
            ma_value = (1.- config.MA_ALPHA) * metrics[name] \
                        + config.MA_ALPHA * self._ma_metrics[f"{name}_ma"]
            self._ma_metrics[f"{name}_ma"] = round(ma_value, 3)
        metrics.update(self._ma_metrics)

    def _update_timer_metrics(self, metrics):
        metrics.update(**Timer.get_times_and_reset())

    def _print_metrics(self, metrics):
        # An estimate of the duration for the episodes that were completed successfully
        duration_done_ma = round(metrics["duration_ma"] / metrics["done_ma"], 3) \
                            if metrics["done_ma"] > .001 else None
        print_metrics = [
            f"episode: {metrics['episode']:4}", 
            f"level: {metrics['difficulty']:2}", 
            f"{100.*metrics['done_ma']:.0f}% success", 
            f"in {duration_done_ma} steps"
        ]
        print("\t".join(print_metrics), end="\r")


class Timer(ContextDecorator):
    """ USAGE:
        with Timer("my_name"):
            <do_things
        
        @Timer.decorate
        def my_func():...

        my_func()

        Timer.get_times_and_reset()
        >>> {"my_name": <time_ms>, "my_func": <time_ms>}
    """
    _TIMES = defaultdict(list)

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, *exc):
        exec_time = (time.time() - self._start_time) * 1000.
        self._TIMES[self._name].append(exec_time)

    @classmethod
    def decorate(cls, func):
        def wrapper_f(*args, **kwargs):
            with cls(func.__name__):
                return func(*args, **kwargs)
        return wrapper_f

    @classmethod
    def get_times_and_reset(cls):
        mean_times = {f"t_{name}": round(sum(times)/len(times), 3) \
                        for name, times in cls._TIMES.items()}
        cls._TIMES = defaultdict(list)
        return mean_times


def product(iterable):
    p = 1
    for n in iterable:
        p *= n
    return p



