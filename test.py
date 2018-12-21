from unityagents import UnityEnvironment
from agents.dqn_agent import DQNAgent
import torch
import numpy as np



def test():
    # set hyperparameters (not really important for running the agent)
    # higher eps. decay rate
    buffer_size = int(1e5)
    batch_size = 64
    gamma = 0.99
    tau = 1e-3
    learning_rate = 5e-4
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.999
    fc1_units = 64
    fc2_units = 64
    q_function_update_fraction=4
    seed = 0

    ############ THE ENVIRONMENT ###############
    env = UnityEnvironment(file_name='Banana_Linux/Banana.x86_64', seed=seed)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # get the number of agents
    num_agents = len(env_info.agents)

    # get the size of the action space
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]


    # initialize agent


    dqn_agent = DQNAgent(name=None,
                         state_size=state_size,
                         action_size=action_size,
                         learning_rate=learning_rate,
                         discount_rate=gamma,
                         eps_start=eps_start,
                         eps_end=eps_end,
                         eps_decay=eps_decay,
                         tau=tau,
                         network_architecture=[fc1_units, fc2_units],
                         experience_replay_buffer_size=buffer_size,
                         experience_replay_buffer_batch_size=batch_size,
                         experience_replay_start_size=3200,
                         q_function_update_fraction=q_function_update_fraction,
                         device='gpu',
                         seed=seed)


    dqn_agent.load_state_dict(torch.load('checkpoint.pth'))

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)

    for i in range(200):
        actions = dqn_agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += rewards
        states = next_states
        if np.any(dones):
            break



if __name__ == '__main__':
    test()
