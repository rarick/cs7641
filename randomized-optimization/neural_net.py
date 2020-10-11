import mlrose_hiive as mlrose
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from functools import wraps

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


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


def rhc_results():
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=0)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(y_test)

    one_hot = OneHotEncoder()

    y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.reshape(-1, 1)).todense()

