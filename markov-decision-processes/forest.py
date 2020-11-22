import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.mdp
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from tqdm import tqdm


def run():
    P, R = ()

    vi = value_iteration(P, R)
    print(vi.run_stats[-1])
    get_performance_discrete(vi.policy, P, R)
    get_performance_continuous(vi.policy)

    pi = policy_iteration(P, R)
    print(pi.run_stats[-1])
    get_performance_discrete(pi.policy, P, R)
    get_performance_continuous(pi.policy)

    ql = q_learning(P, R)
    print(ql.run_stats[-1])
    get_performance_discrete(ql.policy, P, R)
    get_performance_continuous(ql.policy)


def exploration_plots():
    P, R = sample_mountain_car()

    epsilon_mins = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    performance = []
    max_v = []
    mean_v = []
    for epsilon_min in epsilon_mins:
        ql = q_learning(P, R, epsilon_min=epsilon_min)
        performance.append(get_performance_discrete(ql.policy, P, R))
        max_v.append(ql.run_stats[-1]['Max V'])
        mean_v.append(ql.run_stats[-1]['Mean V'])

    plt.figure()
    plt.title('Performance vs $\\epsilon min$')
    plt.xlabel('$\\epsilon min$')
    plt.ylabel('performance')
    plt.scatter(epsilon_mins, performance)
    plt.savefig(f'outputs/epsilon_mins-performance.png')

    plt.figure()
    plt.title('Max V vs $\\epsilon min$')
    plt.xlabel('$\\epsilon min$')
    plt.ylabel('Max V')
    plt.scatter(epsilon_mins, max_v)
    plt.savefig(f'outputs/epsilon_mins-max.png')

    plt.figure()
    plt.title('Mean V vs $\\epsilon min$')
    plt.xlabel('$\\epsilon min$')
    plt.ylabel('Mean V')
    plt.scatter(epsilon_mins, mean_v)
    plt.savefig(f'outputs/epsilon_mins-mean.png')


def value_iteration(P, R):
    np.random.seed(1337)
    print('Value iterating')
    start = timer()
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.99)
    vi.setVerbose()
    vi.run()
    end = timer()
    print(end - start)
    return vi


def policy_iteration(P, R):
    np.random.seed(1337)
    print('Policy iterating')
    start = timer()
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.99, max_iter=200)
    pi.setVerbose()
    pi.run()
    end = timer()
    print(end - start)
    return pi


def q_learning(P, R, epsilon_min=0.2, epsilon_decay=0.9):
    np.random.seed(1337)
    print('Q Learning')
    start = timer()
    ql = mdptoolbox.mdp.QLearning(P, R, 0.99,
                                  n_iter=20000,
                                  alpha_decay=0.999,
                                  alpha_min=0.1,
                                  epsilon_min=epsilon_min,
                                  epsilon_decay=epsilon_decay)
    ql.setVerbose()
    ql.run()
    end = timer()
    print(end - start)
    return ql


def get_performance_discrete(policy, P, R, num_trials=100):
    results = []
    np.random.seed(1337)
    for k in range(num_trials):
        total_reward = 0
        state = np.random.randint(num_positions // 6 * num_velocities, num_positions // 4 * num_velocities)
        for step in range(200):
            action = policy[state]
            total_reward += R[state, action]
            state = np.argmax(P[action, state])

        results.append(total_reward)

    print(f'Average total reward (discrete): {np.mean(results)}')
    return np.mean(results)
