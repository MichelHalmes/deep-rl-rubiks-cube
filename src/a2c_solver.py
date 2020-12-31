
import random
from itertools import count

import torch as T
from torch import optim
import torch.nn.functional as F

import numpy as np

from .network import Network, get_transform_f
from .utils import ReplayBuffer, TrainSchedule, MetricsWriter, Timer, evaluating, Transition
from . import config as cfg


device = T.device("cuda" if T.cuda.is_available() else "cpu")

class A2cCubeSolver(object):

    def __init__(self, cube):
        self._cube = cube
        self._transform_f = get_transform_f(cube.COLORS)
        input_shape = [cube.SIZE, cube.SIZE, 6, len(cube.COLORS)]
        num_actions = len(cube.ACTIONS)
        self._network = Network(input_shape, num_actions, actor_critic=True).to(device)
        self._optimizer = optim.Adam(self._network.parameters(), cfg.LEARNING_RATE)

    def train(self):
        train_writer = MetricsWriter("train")
        eval_writer = MetricsWriter("eval")
        schedule = TrainSchedule()
        for train_cfg in schedule.iter_train_configs():
            episodes = []
            for _ in range(cfg.A2C_NUM_EPISODES):
                self._memory = ReplayBuffer()
                metrics = self._run_episode(train_cfg)
                train_writer.write(metrics)
                episodes.append(self._memory.all())

            loss = 0. # T.tensor(0.)
            for e in episodes:
                disc_return = T.zeros_like(e.reward)
                R = 0.
                for t in reversed(range(len(e.reward))):
                    R = e.reward[t] + cfg.GAMMA * R
                    disc_return[t] = R
                advantage = disc_return - e.state_value
                proba = e.action_probas.gather(1, e.action.reshape(-1, 1))

                log_proba = T.log(proba.squeeze(1))

                actor_loss = (-log_proba * advantage.detach()).mean()
                critic_loss = 0.5 * advantage.pow(2).mean()
                entropy = -(e.action_probas * T.log(e.action_probas)).sum(1).mean()
                # print(actor_loss, critic_loss, entropy)
                # print(proba, entropy)
                loss += actor_loss + critic_loss - .05 * entropy

            self._optimizer.zero_grad()
            loss.backward()
            for param in self._network.parameters():
                param.grad.data.clamp_(-cfg.GRADIENT_CLIP, cfg.GRADIENT_CLIP)
            self._optimizer.step()

            if train_cfg.episode % cfg.EVAL_STEPS == 0:
                metrics = self._evaluate_episode(train_cfg)
                eval_writer.write(metrics)
                print(end="\n")

    def _run_episode(self, train_cfg):
        self._cube.reset(steps=train_cfg.difficulty)

        state = self._get_state_tensor()
        for t in range(train_cfg.max_steps):
            action, action_probas, state_value = self._select_action(state)

            next_state, reward, done = self._apply_action(action)
            self._memory.push(state, action, next_state, reward, done, action_probas, state_value)
            state = next_state

            if done:
                duration = t+1
                break
        else:
            duration = 0

        return {**train_cfg._asdict(), "duration": duration, "done": int(done)}

    @Timer.decorate
    def _select_action(self, state, eval_=False):
        state_batch = T.stack([state])
        if eval_:
            with evaluating(self._network), T.no_grad():
                action_probas, state_value = self._network(state_batch)
        else:
            action_probas, state_value = self._network(state_batch)
        # print(action_probas)
        action_idx = T.multinomial(action_probas[0], 1)[0]
        return action_idx, action_probas[0], state_value[0, 0]


    @Timer.decorate
    def _apply_action(self, action):
        self._cube.step(self._cube.ACTIONS[action.item()])
        done = T.tensor(self._cube.is_done())
        next_state = self._get_state_tensor()
        if not done:
            reward = T.tensor(0.)
        else:
            reward = T.tensor(1.)

        return next_state, reward, done

    def _get_state_tensor(self):
        state = self._cube.get_state()
        state = self._transform_f(state)
        return state

    def _evaluate_episode(self, train_cfg):
        self._cube.reset(steps=train_cfg.difficulty)

        state = self._get_state_tensor()
        for t in range(train_cfg.max_steps):
            action, _, _ = self._select_action(state, eval_=True)  # Always evaluate on Policys
            state, _, done = self._apply_action(action)

            if done:
                duration = t+1
                break
        else:
            duration = 0

        return {**train_cfg._asdict(), "duration": duration, "done": int(done)}

