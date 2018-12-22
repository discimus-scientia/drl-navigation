import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import inspect

from model import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent (object):

    def __init__(self,
                 name,
                 state_size,
                 action_size,
                 learning_rate=5e4,
                 discount_rate=0.99,
                 eps_start=1.0,
                 eps_end=0.01,
                 eps_decay=0.995,
                 tau=1e-3,
                 network_architecture=[64, 64],
                 experience_replay_buffer_size=0,
                 experience_replay_buffer_batch_size=0,
                 experience_replay_start_size=3200,
                 q_function_update_fraction=1,
                 device='gpu',
                 seed=None
    ):

        # get instance parameters
        args = inspect.getfullargspec(self.__init__)[0]
        locals_ = locals()
        self.params = {k: locals_[k] if k in locals_ else None for k in args[1:]}

        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.epsilon = eps_start
        self.tau = tau
        self.network_architecture = network_architecture
        self.experience_replay_buffer_size = experience_replay_buffer_size
        self.experience_replay_buffer_batch_size = experience_replay_buffer_batch_size
        self.experience_replay_start_size = experience_replay_start_size
        self.q_function_update_fraction = q_function_update_fraction # update q-function every update_fraction steps
        if device == 'cpu':
            self.device = torch.device('cpu')
            self.device_string = 'cpu'
        elif device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.device_string = 'gpu'
            else:
                raise ValueError("GPU not available! Choose different device!")

        self.seed = seed

        self.stepcounter = 0
        self.episode = 0

        # if experience replay is to be used, initialize a deque of length
        # experience_replay_buffer_size
        if self.experience_replay_buffer_size != 0:
            self.experience_replay_buffer = ReplayBuffer(
                action_size=self.action_size,
                buffer_size=self.experience_replay_buffer_size,
                batch_size=self.experience_replay_buffer_batch_size,
                device=self.device,
                seed=self.seed
            )
        else:
            raise NotImplementedError("you have to use an experience buffer in the current implementation")
 

        # initialize the Q-tables
        #   for DQN, this is done with Neural Networks
        #   we use target networks, so need 2 networks:
        self.qnetwork_local = QNetwork(state_size, action_size, self.seed,
                                       architecture=self.network_architecture
                                       ).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, self.seed,
                                        architecture=self.network_architecture
                                       ).to(self.device)

        # initialize PyTorch optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)


    def reset(self, newseed_for_agent):
        self.__init__(name=self.name,
                      state_size=self.state_size,
                      action_size=self.action_size,
                      learning_rate=self.learning_rate,
                      discount_rate=self.discount_rate,
                      eps_start=self.eps_start,
                      eps_end=self.eps_end,
                      eps_decay=self.eps_decay,
                      tau=self.tau,
                      network_architecture=self.network_architecture,
                      experience_replay_buffer_size=self.experience_replay_buffer_size,
                      experience_replay_buffer_batch_size=self.experience_replay_buffer_batch_size,
                      experience_replay_start_size=self.experience_replay_start_size,
                      q_function_update_fraction=self.q_function_update_fraction,
                      device=self.device_string,
                      seed=newseed_for_agent
        )


    def act(self, state):
        # epsilon-greedy action selection:
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())

        else:
            #print("selecting action at random")
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """
        If experience replay is used, this funciton saves
        a new experience tuple into replay memory. Every few steps, it calls the self.learn
        method to update the q-function. 
        """
        # First, we increase the stepcounter. This counter counts how many times the learn method
        # is called.
        self.stepcounter += 1

        # ====================== record experience into replay buffer ===========
        self.experience_replay_buffer.add(state, action, reward, next_state, done)

        # ====================== update the Q-function every few steps ===========
        if self.stepcounter % self.q_function_update_fraction == 0:
            # only update if enough examples are present in the replay buffer:
            if len(self.experience_replay_buffer) > self.experience_replay_start_size: #self.experience_replay_buffer_batch_size:
                experiences = self.experience_replay_buffer.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """
        Update the q-function and learn from a given batnch of experiences.
        """
        #print("Entering learn method")
        states, actions, rewards, next_states, dones = experiences


        # double dqn adjustment:
        ddqn = False
        if ddqn:
            # use local network to determine the index of the best action for the next state
            index_of_best_action = self.qnetwork_local(next_states).detach().max(1)[1]

            # use the target network to get the q-value of the action with index_of_best_action
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(
                1, index_of_best_action.unsqueeze(1))

            # Compute Q targets for current states
            Q_targets = rewards + (self.discount_rate * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states 
            Q_targets = rewards + (self.discount_rate * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_network, target_network, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



    def decay_epsilon(self):
        # only linear eps decay implemented atm
        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)

