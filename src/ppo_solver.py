
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

class PpoCubeSolver(object):

    def __init__(self, cube):
        self._cube = cube
        self._transform_f = get_transform_f(cube.COLORS)
        self._network, self._network_old = self._init_networks(cube)
        self._optimizer = optim.Adam(self._network.parameters(), cfg.LEARNING_RATE)

    def _init_networks(self, cube):
        input_shape = [cube.SIZE, cube.SIZE, 6, len(cube.COLORS)]
        num_actions = len(cube.ACTIONS)

        network = Network(input_shape, num_actions, actor_critic=True).to(device)
        network_old = Network(input_shape, num_actions, actor_critic=True).to(device)
        network_old.load_state_dict(network.state_dict())
        network_old.eval()
        return network, network_old

    def train(self):
        train_writer = MetricsWriter("train")
        eval_writer = MetricsWriter("eval")
        schedule = TrainSchedule()
        for train_cfg in schedule.iter_train_configs():
            episodes = ReplayBuffer()
            for _ in range(cfg.PPO_NUM_EPISODES):
                memory = ReplayBuffer()
                metrics = self._run_episode(train_cfg, memory)
                train_writer.write(metrics)
                e = memory.all()
                if not e.done[-1]:
                    _, next_value = self._network_old(T.stack([e.next_state[-1]]))
                else:
                    next_value = 0.
                for t in reversed(range(len(e.reward))):
                    e.target_value[t] = e.reward[t] + cfg.GAMMA * next_value
                    next_value = e.state_value[t]
                    episodes.push(e.state[t], e.action[t], e.next_state[t], e.reward[t],
                                e.done[t], e.action_probas[t], e.state_value[t], e.target_value[t])

            for _ in range(cfg.PPO_NUM_BATCHES):
                batch = episodes.sample(cfg.PPO_BATCH_SIZE)
                new_probas, new_value = self._network(batch.state)
                a = batch.action.reshape(-1, 1)
                ratio = new_probas.gather(1, a) / batch.action_probas.gather(1, a).detach()
                advantage = (batch.target_value - batch.state_value).detach()

                surr1 = ratio * advantage
                surr2 = T.clamp(ratio, 1-cfg.PPO_EPS_CLIP, 1+cfg.PPO_EPS_CLIP) * advantage

                actor_loss = -T.min(surr1, surr2).mean()
                critic_loss = 0.5 * (new_value - batch.target_value.detach()).pow(2).mean()
                entropy = -(new_probas * T.log(new_probas)).sum(1).mean()
                loss = actor_loss + .9 * critic_loss - .04 * entropy

                self._optimizer.zero_grad()
                loss.backward()
                for param in self._network.parameters():
                    param.grad.data.clamp_(-cfg.GRADIENT_CLIP, cfg.GRADIENT_CLIP)
                self._optimizer.step()

            self._network_old.load_state_dict(self._network_old.state_dict())
            if train_cfg.episode % cfg.EVAL_STEPS == 0:
                metrics = self._evaluate_episode(train_cfg)
                eval_writer.write(metrics)
                print(end="\n")

    def _run_episode(self, train_cfg, memory):
        self._cube.reset(steps=train_cfg.difficulty)

        state = self._get_state_tensor()
        for t in range(train_cfg.max_steps):
            action, action_probas, state_value = self._select_action(state)

            next_state, reward, done = self._apply_action(action)
            memory.push(state, action, next_state, reward, done, action_probas, state_value)
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

