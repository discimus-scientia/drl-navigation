from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
import os
import pickle

from agents.dqn_agent import DQNAgent


def train():
    """
    Trains a DQN agent in the Unity Banana environment.
    """

    # set hyperparameters

    #    # udacity dqn baseline: solved after 487 steps
    #    buffer_size = int(1e5)
    #    batch_size = 64
    #    gamma = 0.99
    #    tau = 1e-3
    #    learning_rate = 5e-4
    #    eps_start = 1.0
    #    eps_end = 0.01
    #    eps_decay = 0.995
    #    fc1_units = 64
    #    fc2_units = 64
    #    q_function_update_fraction=4
    #    seed = 0
    #    # larger network in 1st layer
    #    buffer_size = int(1e5)
    #    batch_size = 64
    #    gamma = 0.99
    #    tau = 1e-3
    #    learning_rate = 5e-4
    #    eps_start = 1.0
    #    eps_end = 0.01
    #    eps_decay = 0.995
    #    fc1_units = 128
    #    fc2_units = 64
    #    q_function_update_fraction=4
    #    seed = 0
    #
    #    # smaller network in 1st and 2nd layer
    #    buffer_size = int(1e5)
    #    batch_size = 64
    #    gamma = 0.99
    #    tau = 1e-3
    #    learning_rate = 5e-4
    #    eps_start = 1.0
    #    eps_end = 0.01
    #    eps_decay = 0.995
    #    fc1_units = 32
    #    fc2_units = 16
    #    q_function_update_fraction=4
    #    seed = 0

    #     # higher discount rate
    #     buffer_size = int(1e5)
    #     batch_size = 64
    #     gamma = 0.9999
    #     tau = 1e-3
    #     learning_rate = 5e-4
    #     eps_start = 1.0
    #     eps_end = 0.01
    #     eps_decay = 0.995
    #     fc1_units = 64
    #     fc2_units = 64
    #     q_function_update_fraction=4
    #     seed = 0

    #   # higher eps. decay rate
    #   buffer_size = int(1e5)
    #   batch_size = 64
    #   gamma = 0.99
    #   tau = 1e-3
    #   learning_rate = 5e-4
    #   eps_start = 1.0
    #   eps_end = 0.01
    #   eps_decay = 0.999
    #   fc1_units = 64
    #   fc2_units = 64
    #   q_function_update_fraction=4
    #   seed = 0

    # higher eps. decay rate
    buffer_size = int(1e5)
    batch_size = 64
    gamma = 0.99
    tau = 1e-3
    learning_rate = 5e-4
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.990
    fc1_units = 64
    fc2_units = 64
    q_function_update_fraction=4
    seed = 0

    # use a simple concatenation of all hyperparameters as the experiment name. results are stored in a subfolder
    #   with this name
    experiment_name = "6-smaller_eps_decay-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
        buffer_size, batch_size, gamma, tau, learning_rate, eps_start, eps_end, eps_decay, fc1_units, fc2_units,
        q_function_update_fraction,
        seed)

    # in addition to creating the experiment folder, create subfolders for checkpoints and logs
    if not os.path.isdir(experiment_name):
        os.mkdir(experiment_name)
        os.mkdir(experiment_name+'/checkpoints')
        os.mkdir(experiment_name+'/logs')

    # log the hyperparameters
    with open(experiment_name + '/logs/' + 'hyperparameters.log', 'w') as f:
        print("Buffer size {}\nbatch size {}\ngamma {}\ntau {}\nlearning_rate {}\nfc1-fc2 {}-{}\nq-function_update_fraction {}\nseed {}".format(
            buffer_size, batch_size, gamma, tau, learning_rate, fc1_units, fc2_units, q_function_update_fraction,
            seed), file=f)

    ############ THE ENVIRONMENT ###############
    env = UnityEnvironment(file_name='Banana_Linux/Banana.x86_64', seed=seed)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # get the number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # get the size of the action space
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    dqn_agent = DQNAgent(name=experiment_name,
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



    # run the train loop
    scores_all = train_loop(env=env, brain_name=brain_name,
                            agent=dqn_agent,
                            experiment_name=experiment_name)

    pickle.dump(scores_all, open(experiment_name+'/scores_all.pkg', 'wb'))

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_all) + 1), scores_all)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # finally, close the environment
    env.close()


def train_loop(env, brain_name, agent, experiment_name,
               n_episodes=1000, print_every=100):
    """
    Adopted from the Udacity pendulum DDPG implementation.
    """

    experiment_directory = experiment_name
    checkpoints_directory = experiment_directory + '/checkpoints/'
    log_directory = experiment_directory + '/logs/'

    logfile = open(log_directory + experiment_name + '.log', 'w')

    scores_window = deque(maxlen=print_every)
    scores_all = []
    agent.reset(0)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = 0

        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            score += reward  # update the score (for each agent)
            state = next_state  # roll over states to next time step
            if done:  # exit loop if episode finished
                break

        scores_window.append(score)
        scores_all.append(score)
        # epsilon decay
        agent.decay_epsilon()

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), checkpoints_directory+'checkpoint_local.pth')
            torch.save(agent.qnetwork_target.state_dict(), checkpoints_directory+'checkpoint_target.pth')
            break


    logfile.close()

    return scores_all

if __name__ == '__main__':
    train()