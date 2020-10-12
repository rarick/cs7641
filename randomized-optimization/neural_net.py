import mlrose_hiive as mlrose
import numpy as np
import pandas as pd

from pathlib import Path
from timeit import default_timer as timer
from functools import wraps

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


datasets_path = Path('../datasets')
outputs_path = Path('./outputs')


def fetch_data():
    wine_data = pd.read_csv(datasets_path/'winequality-red.csv')
    x = wine_data.iloc[:,:-1]
    y = wine_data.iloc[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    one_hot = OneHotEncoder()
    y_train = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.values.reshape(-1, 1)).todense()

    return x_train, x_test, y_train, y_test


def run_trials(to_csv=True):
    data = fetch_data()

    def save_results(df, prefix):
        if not to_csv:
            return

        df.to_csv(outputs_path/f'{prefix}_nn.csv')

    print('#### Randomized Hill Climbing ####')
    df = get_results(rhc_nn, data, ['restarts'])
    save_results(df, 'rhc')
    print()

    print('#### Simulated Annealing ####')
    df = get_results(sa_nn, data, ['decay'])
    save_results(df, 'sa')
    print()

    print('#### Genetic Algorithm ####')
    df = get_results(ga_nn, data, ['pop_size'])
    save_results(df, 'ga')
    print()


def get_results(nn_func, data, extra_columns=[]):
    common_col = ['time', 'iterations', 'train_acc', 'test_acc']

    results = nn_func(data)

    df = pd.DataFrame.from_records(
        results, columns=[*extra_columns, *common_col])

    print(df)
    return df


def run_nn(x_train, x_test, y_train, y_test, nn, **kwargs):
    start = timer()
    nn.fit(x_train, y_train)
    time = timer() - start

    fitness_curve = nn.fitness_curve
    iterations = len(fitness_curve)

    pred = nn.predict(x_train)
    train_acc = accuracy_score(y_train, pred)

    pred = nn.predict(x_test)
    test_acc = accuracy_score(y_test, pred)

    return {
        **kwargs,
        'time': time,
        'iterations':iterations,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'fitness_curve': fitness_curve
    }


def rhc_nn(data):
    results = []
    for restarts in [0, 2, 4, 8]:
        nn = mlrose.NeuralNetwork(hidden_nodes=[8, 8],
                                  activation='relu',
                                  algorithm='random_hill_climb',
                                  max_iters=10000,
                                  learning_rate=0.1,
                                  early_stopping=True,
                                  max_attempts=500,
                                  restarts=restarts,
                                  clip_max=5,
                                  random_state=0,
                                  curve=True)

        results.append(run_nn(*data, nn, restarts=restarts))

    return results


def sa_nn(data):
    results = []
    for decay in [0.95, 0.975, 0.99, 0.995]:
        nn = mlrose.NeuralNetwork(hidden_nodes=[8, 8],
                                  activation='relu',
                                  algorithm='simulated_annealing',
                                  max_iters=10000,
                                  learning_rate=0.1,
                                  early_stopping=True,
                                  max_attempts=500,
                                  schedule=mlrose.GeomDecay(decay=decay),
                                  clip_max=5,
                                  random_state=0,
                                  curve=True)

        results.append(run_nn(*data, nn, decay=decay))

    return results


def ga_nn(data):
    results = []
    for pop_size in [64, 128, 256, 512]:
        nn = mlrose.NeuralNetwork(hidden_nodes=[8, 8],
                                  activation='relu',
                                  algorithm='genetic_alg',
                                  max_iters=1000,
                                  learning_rate=0.001,
                                  early_stopping=True,
                                  max_attempts=100,
                                  pop_size=pop_size,
                                  clip_max=5,
                                  random_state=0,
                                  curve=True)

        results.append(run_nn(*data, nn, pop_size=pop_size))

    return results
