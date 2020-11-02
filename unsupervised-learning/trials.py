import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn, sklearn.cluster, sklearn.mixture, sklearn.manifold, sklearn.random_projection

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pathlib import Path


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
    x, y = fetch_data(path)

    one_hot = OneHotEncoder()
    y = one_hot.fit_transform(y.values.reshape(-1, 1)).todense()

    return x, y


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

    get_reduce_cluster(wine, 'wine', 5)
    get_reduce_cluster(heart, 'heart', 18)


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


def run_kmeans(x, k):
    kmeans = sklearn.cluster.KMeans(k)
    return kmeans.fit_predict(x)


def run_exmax(x, k):
    mixture = sklearn.mixture.GaussianMixture(
        k, covariance_type='spherical', max_iter=200, n_init=10)
    return mixture.fit_predict(x)


def split_reduced(reduced):
    x = reduced[:,0]
    y = reduced[:,1]
    return x, y


def run_pca(x):
    reduced = sklearn.decomposition.PCA(n_components=2).fit_transform(x)
    return split_reduced(reduced)


def run_ica(x):
    reduced = sklearn.decomposition.FastICA(n_components=2).fit_transform(x)
    return split_reduced(reduced)


def run_randproj(x):
    reduced = sklearn.random_projection.GaussianRandomProjection(
        n_components=2, random_state=1337).fit_transform(x)
    return split_reduced(reduced)


def run_kernel(x):
    reduced = sklearn.decomposition.KernelPCA(
        n_components=2, kernel='rbf').fit_transform(x)
    return split_reduced(reduced)

