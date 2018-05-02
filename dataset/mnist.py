import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from dataset.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, batch_size=32):
        super(MnistDataset, self).__init__(batch_size=batch_size)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        self.x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        self.x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
        self.y_train_cat = to_categorical(y_train).astype(np.float32)
        self.y_test_cat = to_categorical(y_test).astype(np.float32)

        self.num_classes = 10
        self.shape = (28, 28, 1)

