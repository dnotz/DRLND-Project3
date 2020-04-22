from .ddpg import DDPGAgent


class MADDPG():
    """ Multi-Agent DDPG
    This class is a light-weight wrapper for an DDPG agent to enable multi-agent DDPG training,
    in which all agents are represented by the same agent, incl. same network.
    """

    def __init__(self, state_size, action_size, fc_layer_sizes, buffer_size=30000,
                 batch_size=128, update_interval=16, num_update_steps=1,
                 noise_std=0.2, noise_reduction=0.998, noise_std_min=0.05, warmup=1e4,
                 tau=0.02, gamma=0.99, lr_actor=2e-4, lr_critic=2e-4, seed=0):
        """ Initialize an MADDPG agent object.

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
            tau (float): soft weight update factor
            gamma (float): discount factor
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            seed (int): random seed
        """
        # we represent both / all agents in the environment with the same DDPG agent
        self.agent = DDPGAgent(state_size, action_size, fc_layer_sizes, buffer_size,
                               batch_size, update_interval, num_update_steps, noise_std,
                               noise_reduction, noise_std_min, warmup, tau, gamma, lr_actor, lr_critic, seed)

    def act(self, states, add_noise=True):
        """ Calls the act function for state in states and returns list of actions

        Params
        ======
            states (list of list of float): current states
        """
        return [self.agent.act(state, add_noise) for state in states]

    def step(self, states, actions, rewards, next_states, dones):
        """ Calls the step function for each experience item

        Params
        ======
            states (list of list of float): current states
            actions (list of list of float): actions taken
            rewards (list of float): rewards received
            next_states (list of list of float):  next states
            dones (list of bool): bool whether end of episodes reached
        """
        for s, a, r, n, d in zip(states, actions, rewards, next_states, dones):
            self.agent.step(s, a, r, n, d)
