#!/usr/bin/python3

import numpy as np
import bitstrings as bit
import neural_net as nn


np.random.seed(seed=1337)


def main():
    bit.run_trials(to_csv=True, gen_plots=True)
    # nn.run_trials()


if __name__ == '__main__':
    main()
