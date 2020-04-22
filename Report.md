[//]: # (Image References)

[image1]: doc/reward_plot.png "Rewards"

# Report of Project 3: Collaboration and Competition

In the following, we will describe the MADDPG learning algorithm and our implementation / adaptation. We also give an overview of the used hyperparameters.
Then, we show and discuss our achieved results.
Finally, we give an outlook on future work.

## Learning Algorithm

We chose to implement [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).
Deep Deterministic Policy Gradient (DDPG) is an off-policy algorithm that concurrently learns a Q-function and a policy. It can be regarded as Q-learning for continuous action spaces.
MADDPG is an adaptation and extension of it for multi-agent domains. More specifically, each agent has its own policy that is solely based on its own observations of the enviroment, while each agent's critic is trained on all environment observations and the actions of all agents. Since the critic has access to more information during training it can better guide the agent's learning.
However, in our [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment we control both agents with the same policy and we only need a single actor and single critic.

## Hyperparameters

We train our MADDPG agent for up to `n_episodes = 3500` episodes (far fewer were needed), with up to `max_t = 1000` timesteps each.

For the agent training we use the following hyperparameters:

* **`LR_ACTOR = 1e-3`**
* **`LR_CRITIC = 1e-3`**
* **`GAMMA = 0.99`**
* **`TAU = 1e-3`**
* **`BUFFER_SIZE = 5e5`**
* **`WARMUP_STEPS = 1e4`**
* **`UPDATE_EVERY = 20`**
* **`NUM_UPDATES = 10`**
* **`BATCH_SIZE = 256`**
* **`NOISE_STD = 1.0`**
* **`NOISE_REDUCTION_FACTOR = 0.999`**
* **`NOISE_STD_MIN = 0.05`**

`LR_ACTOR` and `LR_CRITIC` are the respective learning rates of the `Adam` optimizers for updating the actor / critic network weights. We apply clipping to the critic's gradients. After filling the replay buffer for `WARMUP_STEPS` with experiences from random actions, these weights are updated every `UPDATE_EVERY` timesteps and each update consists of `NUM_UPDATES` update steps. For each update step we randomly sample a batch size of `BATCH_SIZE` experiences from the replay buffer (with size `BUFFER_SIZE`).
Rewards are discounted with `GAMMA`. `TAU` is the factor used in the soft network updates.
For exploration, we add random normal noise to actions and scale them with `NOISE_STD`. After each training, this scaling factor is reduced via multiplication with the factor `NOISE_REDUCTION_FACTOR` but a minimum of `NOISE_STD_MIN` shall not be exceeded.

We use fully-connected layers for our network architecture with the following specifications:

* Actor: **`LAYER_SIZES = [128, 128, 2]`**
* Critic: **`LAYER_SIZES = [128, 128, 1]`**

The input to the `Actor` is the 24-dimensional observation. The first two linear layers of the `Actor` are followed by `ReLU` activations, whereas the last one is followed by a `tanh` activation to ensure an output in [-1,1]. Furthermore, we use `batch_norm` before each `FC layer`.

The input to the `Critic` is the (24+2)-dimensional observation+action. The observation is fed through a `batch_norm`, the first `FC layer` and a `ReLU` activations. Then, the output is concatenated with the action and fed through two more `FC layers`, of which the first one is followed by a `ReLU` activation.


## Achieved Scores (Rewards)

We have trained the described MADDP agent for **821 episodes** until it was able to solve the environment, i.e. until the average score reached 0.5 over 100 consecutive episodes.
Below you can see a plot of the scores (rewards) against the number of episodes trained.

![Rewards][image1]

## Ideas for Future Work

We have several ideas for future work to improve the agent's performance. 
It might be beneficial to explore more advanced exploration strategies than simply adding Gaussian noise to the actions. For example, adding noise in the parameter space. Furthermore, prioritized experience replay might facilitate learning, too.
In addition, we soom room for improvement in more advanced network architectures and state pre-processing (e.g. normalization).
Last but not least, it would be interesting to compare the MADDPG performance to implementations of other promising algorithms, such as multi-agent versions of Twin Delayed DDPG (TD3) or Soft Actor-Critic (SAC).