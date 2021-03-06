
import random
from itertools import count

import torch as T
from torch import optim
import torch.nn.functional as F

from .network import Network, get_transform_f
from .utils import ReplayBuffer, TrainSchedule, MetricsWriter, Timer, evaluating
from . import config as cfg


device = T.device("cuda" if T.cuda.is_available() else "cpu")

class DqnCubeSolver(object):

    def __init__(self, cube):
        self._cube = cube
        self._transform_f = get_transform_f(cube.COLORS)
        self._policy_net, self._target_net = self._init_networks(cube)
        self._optimizer = optim.Adam(self._policy_net.parameters(), cfg.LEARNING_RATE)
        self._memory = ReplayBuffer(cfg.DQN_MEMORY_MAX_SIZE)

    def _init_networks(self, cube):
        input_shape = [cube.SIZE, cube.SIZE, 6, len(cube.COLORS)]
        num_actions = len(cube.ACTIONS)

        policy_net = Network(input_shape, num_actions).to(device)
        target_net = Network(input_shape, num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        print(policy_net)
        return policy_net, target_net

    def train(self):
        train_writer = MetricsWriter("train")
        eval_writer = MetricsWriter("eval")
        schedule = TrainSchedule()
        for train_cfg in schedule.iter_train_configs():
            metrics = self._train_episode(train_cfg)
            train_writer.write(metrics)
            if train_cfg.episode % cfg.DQN_TARGET_UPDATE == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())
            if train_cfg.episode % cfg.EVAL_STEPS == 0:
                metrics = self._evaluate_episode(train_cfg)
                eval_writer.write(metrics)
                print(end="\n")

    def _train_episode(self, train_cfg):
        self._cube.reset(steps=train_cfg.difficulty)
        # TODO: maybe add temporal dimension to state

        state = self._get_state_tensor()
        for t in range(train_cfg.max_steps):
            action = self._select_action(state, train_cfg.epsilon)

            next_state, reward, done = self._apply_action(action)
            self._memory.push(state, action, next_state, reward, done)
            state = next_state

            self._optimize_model()
            if done:
                duration = t+1
                break
        else:
            duration = 0

        return {**train_cfg._asdict(), "duration": duration, "done": int(done)}

    @Timer.decorate
    def _select_action(self, state, epsilon):
        if random.random() > epsilon:
            state_batch = T.stack([state])
            with evaluating(self._policy_net), T.no_grad():
                _, action_idx = self._policy_net(state_batch).max(1)
            return action_idx
        else:
            action_idx = T.randint(len(self._cube.ACTIONS), [1])
            return action_idx

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

    @Timer.decorate
    def _optimize_model(self):
        if len(self._memory) < cfg.DQN_BATCH_SIZE:
            return
        batch = self._memory.sample(cfg.DQN_BATCH_SIZE)

        non_final_mask = batch.done.logical_not()
        next_state_value = T.zeros(cfg.DQN_BATCH_SIZE)  # Use 0 value for final state
        next_state_value[non_final_mask] = self._target_net(batch.next_state[non_final_mask]).max(1)[0].detach()  # V(s') = max_a[Q(s',a)]
        expected_state_action_values = (next_state_value * cfg.GAMMA) + batch.reward

        state_action_values = self._policy_net(batch.state).gather(1, batch.action).squeeze(1)  # Q(s,a)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # Huber loss
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-cfg.GRADIENT_CLIP, cfg.GRADIENT_CLIP)
        self._optimizer.step()

    def _get_state_tensor(self):
        state = self._cube.get_state()
        state = self._transform_f(state)
        return state

    def _evaluate_episode(self, train_cfg):
        self._cube.reset(steps=train_cfg.difficulty)

        state = self._get_state_tensor()
        for t in range(train_cfg.max_steps):
            action = self._select_action(state, epsilon=0.)  # Always evaluate on Policys
            state, _, done = self._apply_action(action)

            if done:
                duration = t+1
                break
        else:
            duration = 0

        return {**train_cfg._asdict(), "duration": duration, "done": int(done)}

