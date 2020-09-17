import pandas as pd
import numpy as np
import algorithms as alg

from pathlib import Path


datasets_path = Path('../datasets')


def main():
    wine_data = pd.read_csv(datasets_path/'winequality-red.csv')
    heart_data = pd.read_csv(datasets_path/'heart-failure.csv')

    wine_data = alg.normalize_dataset_inputs(wine_data)
    heart_data = alg.normalize_dataset_inputs(heart_data)

    wine_train, wine_val, wine_test = alg.split_data(wine_data)
    heart_train, heart_val, heart_test = alg.split_data(heart_data)

    print(wine_train.shape)
    print(wine_val.shape)
    print(wine_test.shape)


if __name__ == '__main__':
    main()
