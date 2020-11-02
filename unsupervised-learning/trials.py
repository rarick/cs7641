import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn, sklearn.cluster, sklearn.mixture, sklearn.manifold, \
    sklearn.random_projection, sklearn.neural_network

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pathlib import Path
from sklearn.model_selection import train_test_split


datasets_path = Path('../datasets')
outputs_path = Path('./outputs')


def save_plot(name):
    path = Path(outputs_path/f'{name}.png')
    Path.mkdir(path.parent, parents=True, exist_ok=True)
    plt.savefig(path)


sns.set_theme()

_light = sns.color_palette('pastel', 9)
_dark = sns.color_palette('dark', 9)

_colors = sns.color_palette()


def label_colors(labels):
    colors = _colors
    if len(np.unique(labels)) > 10:
        colors = _light + _dark

    return [colors[label] for label in labels]


def fetch_data(path):
    data = pd.read_csv(path)
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    return x, y


def fetch_data_nn(path):
    data = pd.read_csv(path)
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    one_hot = OneHotEncoder()
    y_train = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.values.reshape(-1, 1)).todense()

    return x_train, x_test, y_train, y_test


def run():
    wine, wine_y = fetch_data(datasets_path/'winequality-red.csv')
    heart, heart_y = fetch_data(datasets_path/'heart-failure.csv')

    # get_kmeans_inertias(wine, 'wine')
    # get_kmeans_inertias(heart, 'heart')

    # get_gaussian_covariance(wine, 'wine')
    # get_gaussian_covariance(heart, 'heart')

    # get_reduce(wine, wine_y, 'wine')
    # get_reduce(heart, heart_y, 'heart')

    # get_cluster_reduce(wine, 'wine', 5)
    # get_cluster_reduce(heart, 'heart', 18)

    # get_reduce_cluster(wine, 'wine', 5)
    # get_reduce_cluster(heart, 'heart', 18)

    ###################
    ## Neural network
    ##########

    wine_train_x, wine_test_x, wine_train_y, wine_test_y = \
        fetch_data_nn(datasets_path/'winequality-red.csv')
    heart_train_x, heart_test_x, heart_train_y, heart_test_y = \
        fetch_data_nn(datasets_path/'heart-failure.csv')

    # get_reduce_train(wine_train_x,
    #                  wine_test_x,
    #                  wine_train_y,
    #                  wine_test_y,
    #                  'wine')

    # get_reduce_train(heart_train_x,
    #                  heart_test_x,
    #                  heart_train_y,
    #                  heart_test_y,
    #                  'heart')

    get_reduce_cluster_train(wine_train_x,
                             wine_test_x,
                             wine_train_y,
                             wine_test_y,
                             'wine',
                             5)

    get_reduce_cluster_train(heart_train_x,
                             heart_test_x,
                             heart_train_y,
                             heart_test_y,
                             'heart',
                             18)


def get_kmeans_inertias(x, name):
    inertias = []
    ks = range(2, 20)
    for k in range(2, 20):
        kmeans = sklearn.cluster.KMeans(k)
        kmeans.fit(x)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.scatter(ks, inertias)
    plt.title(f'k-means inertias ({name})')
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.xticks(ks)

    a, b = np.polyfit(np.log(ks), inertias, 1)
    plt.plot(ks, a*np.log(ks) + b, color=colors[1])

    save_plot(f'performance/inertias-{name}')


def get_gaussian_covariance(x, name):
    covariances = []
    ks = range(2, 20)
    for k in range(2, 20):
        mixture = sklearn.mixture.GaussianMixture(
            k, covariance_type='spherical', max_iter=200, n_init=10)
        mixture.fit(x)
        covariances.append(np.mean(mixture.covariances_))

    plt.figure()
    plt.scatter(ks, covariances)
    plt.title(f'Gaussian mixture mean covariances ({name})')
    plt.xlabel('k')
    plt.ylabel('mean covariance')
    plt.xticks(ks)

    a, b = np.polyfit(np.log(ks), covariances, 1)
    plt.plot(ks, a*np.log(ks) + b, color=_colors[1])

    save_plot(f'performance/covariances-{name}')


def get_reduce(x, y, name):
    reduction_funcs = {'pca': run_pca,
                       'ica': run_ica,
                       'randproj': run_randproj,
                       'kernel': run_kernel}
    unique = np.unique(y).tolist()
    label_map = dict(zip(unique, range(len(unique))))
    labels = [label_map[val] for val in y]

    for reduction_name, reduction_func in reduction_funcs.items():
        reduced_x, reduced_y = reduction_func(x)

        plt.figure()
        plt.scatter(reduced_x, reduced_y, c=label_colors(labels),
                    edgecolors='white', alpha=0.9)
        plt.title(f'{reduction_name} ({name})')
        save_plot(f'reduce/{reduction_name}-{name}')


def get_cluster_reduce(x, name, k):
    cluster_funcs = {'kmeans': run_kmeans,
                     'exmax': run_exmax}
    reduction_funcs = {'pca': run_pca,
                       'ica': run_ica,
                       'randproj': run_randproj,
                       'kernel': run_kernel}

    for cluster_name, cluster_func in cluster_funcs.items():
        for reduction_name, reduction_func in reduction_funcs.items():
            labels = cluster_func(x, k)
            reduced_x, reduced_y = reduction_func(x)

            plt.figure()
            plt.scatter(reduced_x, reduced_y, c=label_colors(labels),
                        edgecolors='white', alpha=0.9)
            plt.title(f'{cluster_name}, {reduction_name} ({name})')
            save_plot(f'cluster_then_reduce/{cluster_name}-{reduction_name}-{name}')


def get_reduce_cluster(x, name, k):
    cluster_funcs = {'kmeans': run_kmeans,
                     'exmax': run_exmax}
    reduction_funcs = {'pca': run_pca,
                       'ica': run_ica,
                       'randproj': run_randproj,
                       'kernel': run_kernel}

    for cluster_name, cluster_func in cluster_funcs.items():
        for reduction_name, reduction_func in reduction_funcs.items():
            reduced_x, reduced_y = reduction_func(x)
            reduced = np.stack([reduced_x, reduced_y], -1)

            labels = cluster_func(reduced, k)

            plt.figure()
            plt.scatter(reduced_x, reduced_y, c=label_colors(labels),
                    edgecolors='white')
            plt.title(f'{reduction_name}, {cluster_name} ({name})')
            save_plot(f'reduce_then_cluster/{reduction_name}-{cluster_name}-{name}')


def get_reduce_train(train_x, test_x, train_y, test_y, name):
    reducers = {'pca': get_pca(),
               'ica': get_ica(),
               'randproj': get_randproj(),
               'kernel': get_kernel()}

    results = []
    for reduction_name, reducer in reducers.items():
        # Train
        reduced_train = reducer.fit_transform(train_x)

        nn = MLPClassifier(hidden_layer_sizes=[512]*4,
                           learning_rate_init=1e-2,
                           early_stopping=True,
                           max_iter=10000)
        nn.fit(reduced_train, train_y)
        train_acc = nn.score(reduced_train, train_y)

        # Test
        reduced_test = reducer.transform(test_x)
        test_acc = nn.score(reduced_test, test_y)

        results.append({'name': f'{name}-{reduction_name}',
                        'train_acc': train_acc,
                        'test_acc': test_acc})

    df = pd.DataFrame.from_records(results, columns=['name', 'train_acc', 'test_acc'])
    print(df)
    df.to_csv(outputs_path/f'reduce-train-{name}.csv')


def get_reduce_cluster_train(train_x, test_x, train_y, test_y, name, k):
    clusters = {'kmeans': get_kmeans(k),
                'exmax': get_exmax(k)}
    reducers = {'pca': get_pca(),
               'ica': get_ica(),
               'randproj': get_randproj(),
               'kernel': get_kernel()}

    results = []
    for cluster_name, cluster in clusters.items():
        for reduction_name, reducer in reducers.items():
            one_hot = OneHotEncoder()

            # Train
            reduced_train = reducer.fit_transform(train_x)
            if cluster_name == 'exmax':
                cluster.fit(reduced_train)
                transformed_train = cluster.predict_proba(reduced_train)
            else:
                transformed_train = cluster.fit_predict(reduced_train)
                transformed_train =  one_hot.fit_transform(
                    transformed_train .reshape(-1, 1)).todense()

            nn = MLPClassifier(hidden_layer_sizes=[256]*3,
                               learning_rate_init=1e-2,
                               early_stopping=True,
                               max_iter=10000)
            nn.fit(transformed_train, train_y)
            train_acc = nn.score(transformed_train, train_y)

            # Test
            reduced_test = reducer.transform(test_x)
            if cluster_name == 'exmax':
                transformed_test = cluster.predict_proba(reduced_test)
            else:
                transformed_test = cluster.predict(reduced_test)
                transformed_test =  one_hot.transform(
                    transformed_test .reshape(-1, 1)).todense()

            test_acc = nn.score(transformed_test, test_y)

            results.append({'name': f'{name}-{reduction_name}-{cluster_name}',
                            'train_acc': train_acc,
                            'test_acc': test_acc})

    df = pd.DataFrame.from_records(results, columns=['name', 'train_acc', 'test_acc'])
    print(df)
    df.to_csv(outputs_path/f'reduce-train-cluster-{name}.csv')


def get_kmeans(k):
    return sklearn.cluster.KMeans(k)


def get_exmax(k):
    return sklearn.mixture.GaussianMixture(k, covariance_type='spherical', max_iter=200, n_init=10)


def run_kmeans(x, k):
    return get_kmeans(k).fit_predict(x)


def run_exmax(x, k):
    return get_exmax(k).fit_predict(x)


def split_reduced(reduced):
    x = reduced[:,0]
    y = reduced[:,1]
    return x, y


def get_pca():
    return sklearn.decomposition.PCA(n_components=2)


def get_ica():
    return sklearn.decomposition.FastICA(n_components=2)


def get_randproj():
    return sklearn.random_projection.GaussianRandomProjection(
        n_components=2, random_state=1337)


def get_kernel():
    return sklearn.decomposition.KernelPCA(
        n_components=2, kernel='rbf')


def run_pca(x):
    reduced = get_pca().fit_transform(x)
    return split_reduced(reduced)


def run_ica(x):
    reduced = get_ica().fit_transform(x)
    return split_reduced(reduced)


def run_randproj(x):
    reduced = get_randproj().fit_transform(x)
    return split_reduced(reduced)


def run_kernel(x):
    reduced = get_kernel().fit_transform(x)
    return split_reduced(reduced)

