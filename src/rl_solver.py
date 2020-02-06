
import random
from itertools import count

import torch as T
from torch import optim
import torch.nn.functional as F

from .dqn import DQN, get_transform_f
from .utils import ReplayMemory, Schedule, MetricsWriter, Timer
from . import config


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

    def train(self):
        writer = MetricsWriter()
        schedule = Schedule()
        for episode in count():
            metrics = self._train_episode(episode, schedule)
            metrics.update(**Timer.get_times_and_reset())
            writer.write(metrics)
            print(episode, metrics["duration"], "\t", end="\r")
            if episode % config.TARGET_UPDATE == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())

    def _train_episode(self, episode, schedule):
        difficulty, epsilon, max_steps = schedule.next_step()
        self._cube.reset(steps=difficulty)
        # TODO: maybe add temporal dimension to state

        state = self._get_state_tensor()
        for t in range(max_steps):
            action = self._select_action(state, epsilon)
            
            next_state, reward, done = self._apply_action(action)
            self._memory.push(state, action, next_state, reward, done)
            state = next_state
            
            self._optimize_model()
            if done:
                duration = t+1
                break
        else:
            duration = 2*max_steps

        return {"_episode": episode, "duration": duration, "difficulty": difficulty, "epsilon": epsilon}

    @Timer.decorate
    def _select_action(self, state, epsilon):
        if random.random() > epsilon:
            state_batch = T.stack([state])
            with T.no_grad():
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
            reward = T.tensor(1000.)

        return next_state, reward, done

    @Timer.decorate
    def _optimize_model(self):
        if len(self._memory) < config.BATCH_SIZE:
            return
        batch = self._memory.sample(config.BATCH_SIZE)
        
        non_final_mask = batch.done.logical_not()
        next_state_value = T.zeros(config.BATCH_SIZE)  # Use 0 value for final state
        next_state_value[non_final_mask] = self._target_net(batch.next_state[non_final_mask]).max(1)[0].detach()  # V(s') = max_a[Q(s',a)]
        expected_state_action_values = (next_state_value * config.GAMMA) + batch.reward

        state_action_values = self._policy_net(batch.state).gather(1, batch.action).squeeze(1)  # Q(s,a)
    
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # Huber loss
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-config.GRADIENT_CLIP, config.GRADIENT_CLIP)
        self._optimizer.step()

    def _get_state_tensor(self):
        state = self._cube.get_state()
        state = self._transform_f(state)
        return state

