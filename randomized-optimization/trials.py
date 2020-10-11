import mlrose_hiive as mlrose
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from functools import wraps


def run_trials():
    print('#### Randomized Hill Climbing ####')
    get_results(rhc_results, ['restarts'])
    print()

    print('#### Simulated Annealing ####')
    get_results(sa_results, ['decay'])
    print()

    print('#### Genetic Algorithm ####')
    get_results(ga_results, ['pop_size'])
    print()

    print('#### MIMIC ####')
    get_results(mimic_results, ['pop_size'])
    print()


def get_results(results_func, extra_columns=[]):
    common_col = ['time', 'iterations', 'fitness']

    knapsack, om, fp = results_func()

    results_to_df = lambda results: \
        pd.DataFrame.from_records(
            results, columns=[*extra_columns, *common_col])

    print('\nKnapsack (MIMIC)')
    knapsack_df = results_to_df(knapsack)
    print(knapsack_df)

    print('\nOne max (Annealing)')
    om_df = results_to_df(om)
    print(om_df)

    print('\nFour peaks (Genetic)')
    fp_df = results_to_df(fp)
    print(fp_df)


_knap_weights = list(range(1, 65))
_knap_values = np.random.choice(range(1, 100), size=64)
def knapsack():
    prob = mlrose.DiscreteOpt(length=64,
                              fitness_fn=mlrose.Knapsack(_knap_weights,
                                                         _knap_values,
                                                         max_weight_pct=0.6))
    prob.set_mimic_fast_mode(True)
    return prob


def one_max():
    prob = mlrose.DiscreteOpt(length=64, fitness_fn=mlrose.OneMax())
    prob.set_mimic_fast_mode(True)
    return prob


def four_peaks():
    prob = mlrose.DiscreteOpt(length=256, fitness_fn=mlrose.FourPeaks(0.1))
    prob.set_mimic_fast_mode(True)
    return prob


def run_test(func, **kwargs):
    start = timer()

    _, fitness, fitness_curve = func(**kwargs)

    time = timer() - start
    iterations = len(fitness_curve)
    return {
        **kwargs,
        'time': time,
        'iterations':iterations,
        'fitness': fitness,
        'fitness_curve': fitness_curve
    }


def rhc_results(restart_nums=[4, 8, 16, 32, 64]):
    knapsack_res = []
    om_res = []
    fp_res = []

    trial = lambda problem: lambda restarts: \
        mlrose.random_hill_climb(problem(),
                                 max_attempts=64,
                                 restarts=restarts,
                                 curve=True)

    knapsack_trial = trial(knapsack)
    om_trial = trial(one_max)
    fp_trial = trial(four_peaks)

    for restarts in restart_nums:
        knapsack_res.append(run_test(knapsack_trial, restarts=restarts))
        om_res.append(run_test(om_trial, restarts=restarts))
        fp_res.append(run_test(fp_trial, restarts=restarts))

    return knapsack_res, om_res, fp_res


def sa_results(decays=[0.95, 0.975, 0.99, 0.995, 0.999]):
    knapsack_res = []
    om_res = []
    fp_res = []

    trial = lambda problem: lambda decay: \
        mlrose.simulated_annealing(problem(),
                                   schedule=mlrose.GeomDecay(decay=decay),
                                   max_attempts=64,
                                   curve=True)

    knapsack_trial = trial(knapsack)
    om_trial = trial(one_max)
    fp_trial = trial(four_peaks)

    for decay in decays:
        knapsack_res.append(run_test(knapsack_trial, decay=decay))
        om_res.append(run_test(om_trial, decay=decay))
        fp_res.append(run_test(fp_trial, decay=decay))

    return knapsack_res, om_res, fp_res


def ga_results(pop_sizes=[64, 128, 256, 512, 1024]):
    knapsack_res = []
    om_res = []
    fp_res = []

    trial = lambda problem: lambda pop_size: \
        mlrose.genetic_alg(problem(),
                           pop_size=pop_size,
                           curve=True)

    knapsack_trial = trial(knapsack)
    om_trial = trial(one_max)
    fp_trial = trial(four_peaks)

    for pop_size in pop_sizes:
        knapsack_res.append(run_test(knapsack_trial, pop_size=pop_size))
        om_res.append(run_test(om_trial, pop_size=pop_size))
        fp_res.append(run_test(fp_trial, pop_size=pop_size))

    return knapsack_res, om_res, fp_res


def mimic_results(pop_sizes=[64, 128, 256, 512, 1024]):
    knapsack_res = []
    om_res = []
    fp_res = []

    trial = lambda problem: lambda pop_size: \
        mlrose.mimic(problem(),
                     pop_size=pop_size,
                     keep_pct=0.4,
                     curve=True)

    knapsack_trial = trial(knapsack)
    om_trial = trial(one_max)
    fp_trial = trial(four_peaks)

    for pop_size in pop_sizes:
        knapsack_res.append(run_test(knapsack_trial, pop_size=pop_size))
        om_res.append(run_test(om_trial, pop_size=pop_size))
        fp_res.append(run_test(fp_trial, pop_size=pop_size))

    return knapsack_res, om_res, fp_res
