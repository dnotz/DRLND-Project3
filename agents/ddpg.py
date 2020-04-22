import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from networks.network import Actor, Critic
from .utils.replay_buffer import ReplayBuffer


class DDPGAgent():
    """ DDPG
    This class implements the DDP algorithm.
    For more information see: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    """

    def __init__(self, state_size, action_size, fc_layer_sizes, buffer_size=30000,
                 batch_size=128, update_interval=16, num_update_steps=1,
                 noise_std=0.2, noise_reduction=0.998, noise_std_min=0.05, warmup=1e4,
                 tau=0.02, gamma=0.99, lr_actor=2e-4, lr_critic=2e-4, seed=0):
        """ Initialize an DDPG agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            fc_layer_sizes (list of int): Layer size of each FC layer
            buffer_size (int): the size of the replay buffer
            batch_size (int): the size of the batches for network updates
            update_interval (int): number of steps between updates
            num_update_steps (int): number of update steps in a row
            noise_std (float): std of Gaussian noise for adding to action
            noise_reduction (float): factor to reduce noise after each update
            noise_std_min (float): the minimum value of noise_std
            tau (float): soft weight update factor
            gamma (float): discount factor
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.num_update_steps = num_update_steps
        self.tau = tau
        self.gamma = gamma
        self.noise_std = noise_std
        self.noise_reduction = noise_reduction
        self.noise_std_min = noise_std_min
        self.warmup = warmup
        self.t = 0

        # seed
        np.random.seed(seed)

        # torch device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # add replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, self.device, seed)

        # define networks, initialize target networks with original networks
        self.actor = Actor(state_size, action_size,
                           fc_layer_sizes, seed=seed).to(self.device)
        self.target_actor = Actor(
            state_size, action_size, fc_layer_sizes, seed=seed).to(self.device)
        self.critic = Critic(state_size, action_size,
                             fc_layer_sizes,  seed=seed).to(self.device)
        self.target_critic = Critic(
            state_size, action_size, fc_layer_sizes, seed=seed).to(self.device)
        self.hard_updates()

        # define optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr_critic, weight_decay=0)

    def act(self, state, add_noise=True):
        """ Computes and returns the action to take

        Params
        ======
            state (list of float): current state
        """
        # input state to actor network in eval mode, get action, add Gaussian noise
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze().cpu().detach().numpy()
        self.actor.train()
        if add_noise:
            action += self.noise_std * np.random.normal(size=self.action_size)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Saves step details and potentially performs network training

        Params
        ======
            state (list of float): current state
            action (list of float): action taken
            reward (float): reward received
            next_state (list of float):  next state
            done (bool): bool whether end of episode reached
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.t += 1
        if self.t >= self.warmup:
            if self.t % self.update_interval == 0:
                if (len(self.replay_buffer) > self.batch_size):
                    self.learn()

    def learn(self):
        """ Performs actor and critic network training """
        for _ in range(self.num_update_steps):
            # sample a random batch of experiences
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size)

            # compute Q targets
            actions_next = self.target_actor(next_states)
            q_targets = rewards + self.gamma * \
                (1 - dones) * self.target_critic(next_states, actions_next)
            q_expected = self.critic(states, actions)

            # compute critic loss, update critic
            critic_loss = F.mse_loss(q_expected, q_targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), .5)  # clip gradients
            self.critic_optimizer.step()

            # update actor
            actions_pred = self.actor(states)
            actor_loss = -self.critic(states, actions_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            self.soft_updates()

        # reduce action sampling noise
        self.noise_std = max(
            self.noise_std * self.noise_reduction, self.noise_std_min)

    def soft_updates(self):
        """ Performs a soft parameter update for target and original networks """
        for target, source in zip([self.target_actor, self.target_critic], [self.actor, self.critic]):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_updates(self):
        """ Performs a hard parameter update for target and original networks """
        for target, source in zip([self.target_actor, self.target_critic], [self.actor, self.critic]):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)
