#!/usr/bin/python3

import algorithms as alg
import pandas as pd
import numpy as np
import itertools

from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm


np.random.seed(seed=1337)
datasets_path = Path('../datasets')


def ex_1(wine_train, wine_val, heart_train, heart_val):
    wine_scores = []
    heart_scores = []
    for alpha in np.linspace(0, 0.05, 20):
        heart_dt = alg.DecisionTree(alpha)
        heart_dt.train(heart_train)
        accuracy = heart_dt.score(heart_val)
        heart_scores.append({'alpha': alpha,
                             'accuracy': accuracy})

        wine_dt = alg.DecisionTree(alpha)
        wine_dt.train(wine_train)
        accuracy = wine_dt.score(wine_val)
        wine_scores.append({'alpha': alpha,
                            'accuracy': accuracy})

    heart_results = pd.DataFrame.from_records(heart_scores)
    wine_results = pd.DataFrame.from_records(wine_scores)

    heart_results.plot.scatter(x='alpha', y='accuracy')
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig('./ex1_heart.png')
    wine_results.plot.scatter(x='alpha', y='accuracy')
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig('./ex1_wine.png')


def ex_2(wine_train, wine_val, heart_train, heart_val):
    wine_scores = []
    heart_scores = []
    combinations = itertools.product([2, 3, 4], [128, 256, 512])
    for hidden_layers, hidden_size in tqdm(combinations, total=9):
        heart_nn = alg.NeuralNetwork(hidden_size, hidden_layers)
        heart_nn.train(heart_train)
        accuracy = heart_nn.score(heart_val)
        heart_scores.append({'hidden_layers': hidden_layers,
                             'hidden_size': hidden_size,
                             'accuracy': accuracy})

        wine_nn = alg.NeuralNetwork(hidden_size, hidden_layers)
        wine_nn.train(wine_train)
        accuracy = wine_nn.score(wine_val)
        wine_scores.append({'hidden_layers': hidden_layers,
                            'hidden_size': hidden_size,
                            'accuracy': accuracy})

    heart_results = pd.DataFrame.from_records(heart_scores)
    wine_results = pd.DataFrame.from_records(wine_scores)

    def nn_plot(df):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots()
        for key, grp in df.groupby(['hidden_layers']):
            ax = grp.plot.scatter(ax=ax,
                                  x='hidden_size',
                                  y='accuracy',
                                  label=f'{key} hidden layers',
                                  color=colors.pop(0))
        plt.show()

    nn_plot(heart_results)
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig('./ex2_heart.png')
    nn_plot(wine_results)
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig('./ex2_wine.png')


def ex_3(wine_train, wine_val, heart_train, heart_val):
    wine_scores = []
    heart_scores = []
    for alpha in np.linspace(0, 0.05, 20):
        heart_dt = alg.Boosting(alpha)
        heart_dt.train(heart_train)
        accuracy = heart_dt.score(heart_val)
        heart_scores.append({'alpha': alpha,
                             'accuracy': accuracy})

        wine_dt = alg.DecisionTree(alpha)
        wine_dt.train(wine_train)
        accuracy = wine_dt.score(wine_val)
        wine_scores.append({'alpha': alpha,
                            'accuracy': accuracy})

    heart_results = pd.DataFrame.from_records(heart_scores)
    wine_results = pd.DataFrame.from_records(wine_scores)

    heart_results.plot.scatter(x='alpha', y='accuracy')
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig('./ex3_heart.png')
    wine_results.plot.scatter(x='alpha', y='accuracy')
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig('./ex3_wine.png')


def ex_4(wine_train, wine_val, heart_train, heart_val):
    wine_scores = []
    heart_scores = []
    for kernel in tqdm(['linear', 'poly', 'rbf']):
        heart_svm = alg.SupportVectorMachine(kernel)
        heart_svm.train(heart_train)
        accuracy = heart_svm.score(heart_val)
        heart_scores.append({'kernel': kernel,
                             'accuracy': accuracy})

        wine_svm = alg.SupportVectorMachine(kernel)
        wine_svm.train(wine_train)
        accuracy = wine_svm.score(wine_val)
        wine_scores.append({'kernel': kernel,
                             'accuracy': accuracy})

    heart_results = pd.DataFrame.from_records(heart_scores)
    wine_results = pd.DataFrame.from_records(wine_scores)

    heart_results.plot.bar(x='kernel', y='accuracy')
    plt.savefig('./ex4_heart.png')
    wine_results.plot.bar(x='kernel', y='accuracy')
    plt.savefig('./ex4_wine.png')


def main():
    wine_data = pd.read_csv(datasets_path/'winequality-red.csv')
    heart_data = pd.read_csv(datasets_path/'heart-failure.csv')

    wine_data = alg.normalize_dataset_inputs(wine_data)
    heart_data = alg.normalize_dataset_inputs(heart_data)

    wine_train, wine_val, wine_test = alg.split_data(wine_data)
    heart_train, heart_val, heart_test = alg.split_data(heart_data)

    # ex_1(wine_train, wine_val, heart_train, heart_val)
    # ex_2(wine_train, wine_val, heart_train, heart_val)
    # ex_3(wine_train, wine_val, heart_train, heart_val)
    # ex_4(wine_train, wine_val, heart_train, heart_val)
    ex_5(wine_train, wine_val, heart_train, heart_val)


if __name__ == '__main__':
    main()
