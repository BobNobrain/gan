import numpy as np


class Dataset:
    def __init__(self, batch_size=32):
        self.x_train = None
        self.y_train_cat = None
        self.x_test = None
        self.y_test_cat = None
        self.num_classes = 0
        self.shape = (28, 28)
        self.batch_size = batch_size

    def batch_iterator(self, x, y):
        n_batches = x.shape[0] // self.batch_size
        while True:
            for i in range(n_batches):
                # TODO: extract this to variables
                # sample_index_start = self.batch_size * i
                # sample_index_end = self.batch_size * (i + 1)
                yield x[
                    self.batch_size * i: self.batch_size * (i + 1)
                ], y[
                    self.batch_size * i: self.batch_size * (i + 1)
                ]
            idxs = np.random.permutation(y.shape[0])
            x = x[idxs]
            y = y[idxs]

    def train_batch_iterator(self):
        return self.batch_iterator(self.x_train, self.y_train_cat)

    def test_batch_iterator(self):
        return self.batch_iterator(self.x_test, self.y_test_cat)
