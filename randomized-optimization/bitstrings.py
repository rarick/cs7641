import mlrose_hiive as mlrose
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from functools import wraps


outputs_path = Path('./outputs')
outputs_path.mkdir(parents=True, exist_ok=True)


def run_trials(to_csv=True, gen_plots=True):

    def save_results(knap, om, fp, prefix):
        if not to_csv:
            return
        knap.to_csv(outputs_path/f'{prefix}_knap.csv')
        om.to_csv(outputs_path/f'{prefix}_om.csv')
        fp.to_csv(outputs_path/f'{prefix}_fp.csv')

    print('#### Randomized Hill Climbing ####')
    results = get_results(rhc_results, ['restarts'])
    save_results(*results, 'rhc')
    print()

    print('#### Simulated Annealing ####')
    results = get_results(sa_results, ['decay'])
    save_results(*results, 'sa')
    print()

    print('#### Genetic Algorithm ####')
    results = get_results(ga_results, ['pop_size'])
    save_results(*results, 'ga')
    print()

    print('#### MIMIC ####')
    results = get_results(mimic_results, ['pop_size'])
    save_results(*results, 'mimic')
    print()

    if gen_plots:
        gen_report_plots()


def gen_report_plots():
    pass



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

    return knapsack_df, om_df, fp_df


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
    prob = mlrose.DiscreteOpt(length=512, fitness_fn=mlrose.FourPeaks())
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
                                 random_state=0,
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
                                   random_state=0,
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
                           random_state=0,
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
                     random_state=0,
                     curve=True)

    knapsack_trial = trial(knapsack)
    om_trial = trial(one_max)
    fp_trial = trial(four_peaks)

    for pop_size in pop_sizes:
        knapsack_res.append(run_test(knapsack_trial, pop_size=pop_size))
        om_res.append(run_test(om_trial, pop_size=pop_size))
        fp_res.append(run_test(fp_trial, pop_size=pop_size))

    return knapsack_res, om_res, fp_res
