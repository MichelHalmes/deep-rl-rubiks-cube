import logging
import sys
import random
from itertools import count

import torch as T
from torch import optim
import torch.nn.functional as F

from dqn import DQN, get_transform_f
from cube_wrapper import MyCube
from utils import ReplayMemory, Transition

EPS = .1
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 10
MAX_STEPS = 10
GRADIENT_CLIP = .1

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class RlCubeSolver(object):

    def __init__(self, cube):
        self._cube = cube
        self._transform_f = get_transform_f(cube.COLORS)
        self._policy_net, self._target_net = self._init_networks(cube)
        self._optimizer = optim.Adam(self._policy_net.parameters())
        self._memory = ReplayMemory()

    def _init_networks(self, cube):
        input_shape = [cube.SIZE, cube.SIZE, 6, len(cube.COLORS)]
        output_size = len(cube.ACTIONS)

        policy_net = DQN(input_shape, output_size).to(device)
        target_net = DQN(input_shape, output_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        return policy_net, target_net

    def _select_action(self, state):
        if random.random() > EPS:
            state_batch = T.stack([state])
            with T.no_grad():
                _, action_idx = self._policy_net(state_batch).max(1)
            return action_idx
        else:
            action_idx = T.randint(len(self._cube.ACTIONS), [1])
            return action_idx

    def _optimize_model(self):
        if len(self._memory) < BATCH_SIZE:
            return
        batch = self._memory.sample(BATCH_SIZE)
        
        non_final_mask = batch.done.logical_not()
        next_state_value = T.zeros(BATCH_SIZE)  # Use 0 value for final state
        next_state_value[non_final_mask] = self._target_net(batch.next_state[non_final_mask]).max(1)[0].detach()  # V(s') = max_a[Q(s',a)]
        expected_state_action_values = (next_state_value * GAMMA) + batch.reward

        state_action_values = self._policy_net(batch.state).gather(1, batch.action).squeeze(1)  # Q(s,a)
    
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # Huber loss
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-GRADIENT_CLIP, GRADIENT_CLIP)
        self._optimizer.step()

    def _get_state_tensor(self):
        state = self._cube.get_state()
        state = self._transform_f(state)
        return state

    def _train_episode(self, episode_idx):
        self._cube.reset(steps=1)
        # TODO: maybe add temporal dimension to state

        state = self._get_state_tensor()
        for t in range(MAX_STEPS):
            action = self._select_action(state)
            self._cube.step(self._cube.ACTIONS[action.item()])
            done = T.tensor(self._cube.is_done())

            next_state = self._get_state_tensor()
            if not done:
                reward = T.tensor(0.)
            else:
                reward = T.tensor(1000.)

            self._memory.push(state, action, next_state, reward, done)
            state = next_state

            self._optimize_model()
            if done:
                return t+1

        return 2*MAX_STEPS

    def train(self):
        for episode_idx in count():
            duration = self._train_episode(episode_idx)
            print(episode_idx, duration)
            if episode_idx % TARGET_UPDATE == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())


def main():
    cube = MyCube()
    rl_solver = RlCubeSolver(cube)
    rl_solver.train()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    main()
    



