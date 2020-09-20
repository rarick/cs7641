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


def decision_tree(wine_train, wine_test, heart_train, heart_test):
    wine_train_scores = []
    heart_train_scores = []
    wine_test_scores = []
    heart_test_scores = []
    for alpha in np.linspace(0, 0.05, 20):
        heart_dt = alg.DecisionTree(alpha)
        heart_dt.train(heart_train)
        accuracy = heart_dt.score(heart_train)
        heart_train_scores.append({'alpha': alpha,
                                   'accuracy': accuracy})
        accuracy = heart_dt.score(heart_test)
        heart_test_scores.append({'alpha': alpha,
                                  'accuracy': accuracy})

        wine_dt = alg.DecisionTree(alpha)
        wine_dt.train(wine_train)
        accuracy = wine_dt.score(wine_train)
        wine_train_scores.append({'alpha': alpha,
                                  'accuracy': accuracy})
        accuracy = wine_dt.score(wine_test)
        wine_test_scores.append({'alpha': alpha,
                                 'accuracy': accuracy})

    heart_train_results = pd.DataFrame.from_records(heart_train_scores)
    wine_train_results = pd.DataFrame.from_records(wine_train_scores)

    heart_test_results = pd.DataFrame.from_records(heart_test_scores)
    wine_test_results = pd.DataFrame.from_records(wine_test_scores)

    colors = alg.ColorPopper()
    ax = heart_train_results.plot.scatter(x='alpha',
                                          y='accuracy',
                                          color=colors.pop(),
                                          label='train')
    heart_test_results.plot.scatter(x='alpha',
                                    y='accuracy',
                                    ax=ax,
                                    color=colors.pop(),
                                    label='test')
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Heart Failure Decision Tree Train/Test Accurarcy')
    plt.legend(loc='lower left')
    plt.savefig('./dt_heart.png')

    colors = alg.ColorPopper()
    ax = wine_train_results.plot.scatter(x='alpha',
                                         y='accuracy',
                                         color=colors.pop(),
                                         label='train')
    wine_test_results.plot.scatter(x='alpha',
                                   y='accuracy',
                                   ax=ax,
                                   color=colors.pop(),
                                   label='test')
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Wine Quality Decision Tree Train/Test Accurarcy')
    plt.legend(loc='lower left')
    plt.savefig('./dt_wine.png')


def neural_network(wine_train, wine_test, heart_train, heart_test):
    wine_train_scores = []
    heart_train_scores = []
    wine_test_scores = []
    heart_test_scores = []
    combinations = itertools.product([2, 3, 4], [128, 256, 512])
    for hidden_layers, hidden_size in tqdm(combinations, total=9):
        heart_nn = alg.NeuralNetwork(hidden_size, hidden_layers)
        heart_nn.train(heart_train)
        accuracy = heart_nn.score(heart_train)
        heart_train_scores.append({'hidden_layers': hidden_layers,
                                   'hidden_size': hidden_size,
                                   'accuracy': accuracy})
        accuracy = heart_nn.score(heart_test)
        heart_test_scores.append({'hidden_layers': hidden_layers,
                                  'hidden_size': hidden_size,
                                  'accuracy': accuracy})

        wine_nn = alg.NeuralNetwork(hidden_size, hidden_layers)
        wine_nn.train(wine_train)
        accuracy = wine_nn.score(wine_train)
        wine_train_scores.append({'hidden_layers': hidden_layers,
                                  'hidden_size': hidden_size,
                                  'accuracy': accuracy})
        accuracy = wine_nn.score(wine_test)
        wine_test_scores.append({'hidden_layers': hidden_layers,
                                 'hidden_size': hidden_size,
                                 'accuracy': accuracy})

    heart_train_results = pd.DataFrame.from_records(heart_train_scores)
    wine_train_results = pd.DataFrame.from_records(wine_train_scores)

    heart_test_results = pd.DataFrame.from_records(heart_test_scores)
    wine_test_results = pd.DataFrame.from_records(wine_test_scores)

    def nn_plot(df):
        colors = alg.ColorPopper()
        fig, ax = plt.subplots()
        for key, grp in df.groupby(['hidden_layers']):
            ax = grp.plot.scatter(ax=ax,
                                  x='hidden_size',
                                  y='accuracy',
                                  label=f'{key} hidden layers',
                                  color=colors.pop())
        plt.legend(loc='lower left')
        plt.show()

    nn_plot(heart_train_results)
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Heart Failure Neural Network Training Accuracy')
    plt.savefig('./nn_heart_train.png')
    nn_plot(wine_train_results)
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Wine Quality Neural Network Training Accuracy')
    plt.savefig('./nn_wine_train.png')

    nn_plot(heart_test_results)
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Heart Failure Neural Network Test Accuracy')
    plt.savefig('./nn_heart_test.png')
    nn_plot(wine_test_results)
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Wine Quality Neural Network Test Accuracy')
    plt.savefig('./nn_wine_test.png')


def boosting(wine_train, wine_test, heart_train, heart_test):
    wine_train_scores = []
    heart_train_scores = []
    wine_test_scores = []
    heart_test_scores = []
    for alpha in np.linspace(0, 0.05, 20):
        heart_dt = alg.Boosting(alpha)
        heart_dt.train(heart_train)
        accuracy = heart_dt.score(heart_train)
        heart_train_scores.append({'alpha': alpha,
                                   'accuracy': accuracy})
        accuracy = heart_dt.score(heart_test)
        heart_test_scores.append({'alpha': alpha,
                                  'accuracy': accuracy})

        wine_dt = alg.Boosting(alpha)
        wine_dt.train(wine_train)
        accuracy = wine_dt.score(wine_train)
        wine_train_scores.append({'alpha': alpha,
                                  'accuracy': accuracy})
        accuracy = wine_dt.score(wine_test)
        wine_test_scores.append({'alpha': alpha,
                                 'accuracy': accuracy})

    heart_train_results = pd.DataFrame.from_records(heart_train_scores)
    wine_train_results = pd.DataFrame.from_records(wine_train_scores)

    heart_test_results = pd.DataFrame.from_records(heart_test_scores)
    wine_test_results = pd.DataFrame.from_records(wine_test_scores)

    colors = alg.ColorPopper()
    ax = heart_train_results.plot.scatter(x='alpha',
                                          y='accuracy',
                                          color=colors.pop(),
                                          label='train')
    heart_test_results.plot.scatter(x='alpha',
                                    y='accuracy',
                                    ax=ax,
                                    color=colors.pop(),
                                    label='test')
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Heart Failure Random Forest Train/Test Accurarcy')
    plt.legend(loc='lower left')
    plt.savefig('./boost_heart.png')

    colors = alg.ColorPopper()
    ax = wine_train_results.plot.scatter(x='alpha',
                                         y='accuracy',
                                         color=colors.pop(),
                                         label='train')
    wine_test_results.plot.scatter(x='alpha',
                                   y='accuracy',
                                   ax=ax,
                                   color=colors.pop(),
                                   label='test')
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Wine Quality Random Forest Train/Test Accurarcy')
    plt.legend(loc='lower left')
    plt.savefig('./boost_wine.png')


def svm(wine_train, wine_test, heart_train, heart_test):
    wine_train_scores = []
    heart_train_scores = []
    wine_test_scores = []
    heart_test_scores = []
    for kernel in tqdm(['linear', 'poly', 'rbf']):
        heart_svm = alg.SupportVectorMachine(kernel)
        heart_svm.train(heart_train)
        accuracy = heart_svm.score(heart_train)
        heart_train_scores.append({'kernel': kernel,
                                   'accuracy': accuracy})
        accuracy = heart_svm.score(heart_test)
        heart_test_scores.append({'kernel': kernel,
                                  'accuracy': accuracy})

        wine_svm = alg.SupportVectorMachine(kernel)
        wine_svm.train(wine_train)
        accuracy = wine_svm.score(wine_train)
        wine_train_scores.append({'kernel': kernel,
                                  'accuracy': accuracy})
        accuracy = wine_svm.score(wine_test)
        wine_test_scores.append({'kernel': kernel,
                                 'accuracy': accuracy})

    heart_train_results = pd.DataFrame.from_records(heart_train_scores)
    wine_train_results = pd.DataFrame.from_records(wine_train_scores)

    heart_test_results = pd.DataFrame.from_records(heart_test_scores)
    wine_test_results = pd.DataFrame.from_records(wine_test_scores)

    colors = alg.ColorPopper()
    ax = heart_train_results.plot.bar(x='kernel',
                                          y='accuracy',
                                          color=colors.pop(),
                                          label='train')
    heart_test_results.plot.bar(x='kernel',
                                    y='accuracy',
                                    ax=ax,
                                    color=colors.pop(),
                                    label='test')
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Heart Failure SVM Train/Test Accurarcy')
    plt.legend(loc='lower left')
    plt.savefig('./svm_heart.png')

    colors = alg.ColorPopper()
    ax = wine_train_results.plot.bar(x='kernel',
                                     y='accuracy',
                                     color=colors.pop(),
                                     label='train')
    wine_test_results.plot.bar(x='kernel',
                               y='accuracy',
                               ax=ax,
                               color=colors.pop(),
                               label='test')
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Wine Quality SVM Train/Test Accurarcy')
    plt.legend(loc='lower left')
    plt.savefig('./svm_wine.png')


def knn(wine_train, wine_test, heart_train, heart_test):
    wine_train_scores = []
    heart_train_scores = []
    wine_test_scores = []
    heart_test_scores = []
    for k in tqdm(range(1, 19, 2)):
        heart_knn = alg.NearestNeighbors(k)
        heart_knn.train(heart_train)
        accuracy = heart_knn.score(heart_train)
        heart_train_scores.append({'k': k,
                                   'accuracy': accuracy})
        accuracy = heart_knn.score(heart_test)
        heart_test_scores.append({'k': k,
                                  'accuracy': accuracy})

        wine_knn = alg.NearestNeighbors(k)
        wine_knn.train(wine_train)
        accuracy = wine_knn.score(wine_train)
        wine_train_scores.append({'k': k,
                                  'accuracy': accuracy})
        accuracy = wine_knn.score(wine_test)
        wine_test_scores.append({'k': k,
                                 'accuracy': accuracy})

    heart_train_results = pd.DataFrame.from_records(heart_train_scores)
    wine_train_results = pd.DataFrame.from_records(wine_train_scores)

    heart_test_results = pd.DataFrame.from_records(heart_test_scores)
    wine_test_results = pd.DataFrame.from_records(wine_test_scores)

    colors = alg.ColorPopper()
    ax = heart_train_results.plot.scatter(x='k',
                                          y='accuracy',
                                          color=colors.pop(),
                                          label='train')
    heart_test_results.plot.scatter(x='k',
                                    y='accuracy',
                                    ax=ax,
                                    color=colors.pop(),
                                    label='test')
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Heart Failure KNN Train/Test Accurarcy')
    plt.legend(loc='lower left')
    plt.savefig('./knn_heart.png')

    colors = alg.ColorPopper()
    ax = wine_train_results.plot.scatter(x='k',
                                         y='accuracy',
                                         color=colors.pop(),
                                         label='train')
    wine_test_results.plot.scatter(x='k',
                                   y='accuracy',
                                   ax=ax,
                                   color=colors.pop(),
                                   label='test')
    plt.ylim(bottom=0, top=1)
    plt.grid()
    plt.title('Wine Quality KNN Train/Test Accurarcy')
    plt.legend(loc='lower left')
    plt.savefig('./knn_wine.png')


def main():
    wine_data = pd.read_csv(datasets_path/'winequality-red.csv')
    heart_data = pd.read_csv(datasets_path/'heart-failure.csv')

    wine_data = alg.normalize_dataset_inputs(wine_data)
    heart_data = alg.normalize_dataset_inputs(heart_data)

    wine_train, wine_test = alg.split_data(wine_data)
    heart_train, heart_test = alg.split_data(heart_data)

    decision_tree(wine_train, wine_test, heart_train, heart_test)
    neural_network(wine_train, wine_test, heart_train, heart_test)
    boosting(wine_train, wine_test, heart_train, heart_test)
    svm(wine_train, wine_test, heart_train, heart_test)
    knn(wine_train, wine_test, heart_train, heart_test)


if __name__ == '__main__':
    main()
