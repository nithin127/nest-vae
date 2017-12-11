import h5py
import numpy as np
import random

class FactorSampler(object):

    def __init__(self, filename, batch_size, max_iter=None):
        self.filename = filename
        self.batch_size = batch_size
        self.max_iter = max_iter

        self.factors = h5py.File(filename, 'r')
        self.num_factors = len(self.factors['factor'])
        self._iter = 0

    def __iter__(self):
        while (self.max_iter is None) or (self._iter < self.max_iter):
            for index in np.random.permutation(self.num_factors):
                factor_type = self.factors['factor/{0:d}'.format(index)]
                factor_value = random.choice(list(factor_type))
                factor = self.factors['factor/{0:d}/{1}'.format(
                    index, factor_value)]

                indices = np.random.choice(len(factor), self.batch_size)
                batch = [factor[index] for index in indices]
                self._iter += 1
                yield batch

    def __len__(self):
        return self.max_iter
