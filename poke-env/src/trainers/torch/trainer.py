import math
import random
from copy import deepcopy
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from gym import Env
from tqdm import tqdm

from .memory import ReplayMemory, Transition


EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000


class TorchTrainer:
    """ Adapted from pytorch DQN tutorial.

    References:
    [1] Reinforcement Learning (DQN) Tutorial. 
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(
        self, 
        model: nn.Module,
        env: Env,
        memory: ReplayMemory,
        gamma: float = 0.99,
        lr: float = 1e-4,
        tau: float = 0.005,
        batch_size: int = 128,
        device: str = 'cpu',
    ):
        self.policy_net = deepcopy(model)
        self.target_net = deepcopy(model)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.env = env
        self.memory = memory
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        self.steps_done = 0
        self.episode_durations = []

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)


    def select_action(self, state):
        """ Will select an action accordingly to an epsilon greedy policy. 
        Simply put, we'll sometimes use our model for choosing the action, and 
        sometimes we'll just sample one uniformly. The probability of choosing 
        a random action will start at EPS_START and will decay exponentially 
        towards EPS_END. EPS_DECAY controls the rate of the decay.
        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)


    def optimize_model(self):
        """ Performs a single step of the optimization. It first samples a 
        batch, concatenates all the tensors into a single one, computes
        Q(s_t, a_t) and V(s_{t+1}) = max_a Q(s_{t+1}, a), and combines them 
        into our loss. 
        """

        if len(self.memory) < self.batch_size:
            return 
        
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def train_model(self):
        
        if torch.cuda.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        # if is_ipython:
        #     if not show_result:
        #         display.display(plt.gcf())
        #         display.clear_output(wait=True)
        #     else:
        #         display.display(plt.gcf())