import gym
import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.mdp
import numpy as np

from timeit import default_timer as timer
from tqdm import tqdm


# num_positions=128
# num_velocities=128
# num_actions=8

num_positions=63
num_velocities=63
num_actions=8


def run():
    P, R = sample_mountain_car()

    vi = value_iteration(P, R)
    print(vi.run_stats[-1])
    get_performance(vi.policy)

    pi = policy_iteration(P, R)
    print(pi.run_stats[-1])
    get_performance(pi.policy)

    ql = q_learning(P, R)
    print(ql.run_stats[-1])
    get_performance(ql.policy)


# def sample_mountain_car(num_positions=127, num_velocities=63, num_actions=15):
def sample_mountain_car():
    num_states = num_positions*num_velocities

    env = gym.make('MountainCarContinuous-v0')
    env.reset()

    action_space = env.action_space
    observation_space = env.observation_space

    position_min, velocity_min = observation_space.low
    position_max, velocity_max = observation_space.high

    action_min, action_max = (action_space.low[0], action_space.high[0])

    positions = np.linspace(position_min, position_max, num_positions)
    velocities = np.linspace(velocity_min, velocity_max, num_velocities)
    action_space = np.linspace(action_min, action_max, num_actions)

    states = []
    actions = []
    rewards = []
    state_primes = []
    for position in tqdm(positions):
        for velocity in velocities:
            for action in action_space:
                env.state[:] = (position, velocity)
                new_state, reward, done, _ = env.step([action])

                states.append((position, velocity))
                actions.append(action)
                rewards.append(reward)
                state_primes.append(new_state)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    state_primes = np.array(state_primes)

    action_indices = np.digitize(actions, action_space, right=True)

    position_indices = np.digitize(states[:,0], positions, right=True)
    velocity_indices = np.digitize(states[:,1], velocities, right=True)
    state_indices = num_velocities*position_indices + velocity_indices

    position_prime_indices = np.digitize(state_primes[:,0], positions, right=True)
    velocity_prime_indices = np.digitize(state_primes[:,1], velocities, right=True)
    state_prime_indices = num_velocities*position_prime_indices + velocity_prime_indices

    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))

    for s, a, r, sp in zip(state_indices,
                           action_indices,
                           rewards,
                           state_prime_indices):
        P[a, s, sp] += 1
        R[s, a] = r

    assert (P.sum(axis=2) == 1).all()

    return P, R


def value_iteration(P, R):
    print('Value iterating')
    start = timer()
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    vi.setVerbose()
    vi.run()
    end = timer()
    print(end - start)
    return vi


def policy_iteration(P, R):
    print('Policy iterating')
    start = timer()
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    pi.setVerbose()
    pi.run()
    end = timer()
    print(end - start)
    return pi


def q_learning(P, R):
    print('Q Learning')
    start = timer()
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, alpha_decay=0.999, alpha_min=0.1, epsilon_min=0.2, epsilon_decay=0.999)
    ql.setVerbose()
    ql.run()
    end = timer()
    print(end - start)
    return ql


def get_performance(policy, num_trials=10):
    env = gym.make('MountainCarContinuous-v0')
    env.reset()

    num_states = num_positions*num_velocities
    assert len(policy) == num_states

    action_space = env.action_space
    observation_space = env.observation_space

    position_min, velocity_min = observation_space.low
    position_max, velocity_max = observation_space.high

    action_min, action_max = (action_space.low[0], action_space.high[0])

    positions = np.linspace(position_min, position_max, num_positions)
    velocities = np.linspace(velocity_min, velocity_max, num_velocities)
    action_space = np.linspace(action_min, action_max, num_actions)

    trial_data = []
    for k in range(num_trials):
        env = gym.make('MountainCarContinuous-v0')
        state = env.reset()
        total_reward = 0
        while True:
            position_index = np.digitize(state[0], positions, right=True)
            velocity_index = np.digitize(state[1], velocities, right=True)
            state_index = num_velocities*position_index + velocity_index

            action_index = policy[state_index]
            action = action_space[action_index]
            print(f'{(state, action)}')
            state, reward, done, _ = env.step([action])
            total_reward += reward
            if done:
                break

        trial_data.append(total_reward)

    print(trial_data)
