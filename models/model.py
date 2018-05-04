import tensorflow as tf

from dataset.dataset import Dataset
from model_train_listener import ModelTrainListener


class Model:
    def __init__(
            self,
            latent_dim=2,
            batches_per_period=20,
            dirname='./weights/',
            save_period=100
    ):
        self.sess = None
        self.latent_dim = latent_dim
        self.batches_per_period = batches_per_period
        self.dirname = dirname
        self.save_period = save_period
        self.listeners = []

    def train(self, dataset: Dataset, epochs=5000):
        pass

    def init_model(self, num_classes, shape, print_summary=False):
        pass

    def feed(self, z_sample, input_lbl):
        pass

    def add_listener(self, listener: ModelTrainListener):
        self.listeners.append(listener)

    def save_weights(self, step):
        saver = tf.train.Saver()
        saver.save(self.sess, self.dirname + 'n', global_step=step)

    def load_weights(self):
        print('Loading weights...')
        saver = tf.train.Saver()
        lp = tf.train.latest_checkpoint(self.dirname)
        saver.restore(self.sess, lp)
        print("Model loaded from: {}".format(lp))
