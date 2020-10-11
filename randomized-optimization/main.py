#!/usr/bin/python3

import numpy as np
import pandas as pd
import bitstrings as bit
import neural_net as nn

from matplotlib import pyplot as plt


np.random.seed(seed=1337)


def main():
    # bit.run_trials()
    nn.run_trials()


if __name__ == '__main__':
    main()
