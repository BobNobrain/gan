import numpy as np

from keras.layers import Dropout, Reshape, Flatten, RepeatVector
from keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model

from keras import backend as keras_backend
import tensorflow as tf

from model_train_listener import ModelTrainListener
from dataset.dataset import Dataset


class GANModel:
    def __init__(
            self,
            # num_classes,
            latent_dim=2,
            dropout_rate=0.3,
            k_step=5,
            batches_per_period=20
    ):
        self.sess = None
        # self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.k_step = k_step  # Количество шагов, которые могут делать дискриминатор и генератор во внутреннем цикле
        self.batches_per_period = batches_per_period
        self.dropout_rate = dropout_rate

        self.L_gen = None
        self.L_dis = None

        self.step_gen = None
        self.step_dis = None

        self.train_batches_it = None
        self.test_batches_it = None

        self.generated_z = None
        self.gan = None

        self.listeners = []

        # self.x_ = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='image')
        # self.y_ = tf.placeholder(tf.float32, shape=(None, num_classes), name='labels')
        # self.z_ = tf.placeholder(tf.float32, shape=(None, latent_dim), name='z')
        #
        # self.img = Input(tensor=self.x_)
        # self.lbl = Input(tensor=self.y_)
        # self.z = Input(tensor=self.z_)
        self.x_ = None
        self.y_ = None
        self.z_ = None
        self.img = None
        self.lbl = None
        self.z = None

    def init_session(self, force=False):
        if (not self.sess) or force:
            self.sess = tf.Session()
            keras_backend.set_session(self.sess)

    def init_vectors(self, num_classes, shape):
        x, y, channels = shape
        self.x_ = tf.placeholder(tf.float32, shape=(None, x, y, channels), name='image')
        self.y_ = tf.placeholder(tf.float32, shape=(None, num_classes), name='labels')
        self.z_ = tf.placeholder(tf.float32, shape=(None, self.latent_dim), name='z')

        self.img = Input(tensor=self.x_)
        self.lbl = Input(tensor=self.y_)
        self.z = Input(tensor=self.z_)

    @staticmethod
    def gen_batch(x, y, batch_size):
        n_batches = x.shape[0] // batch_size
        while True:
            for i in range(n_batches):
                yield x[batch_size*i: batch_size*(i+1)], y[batch_size*i: batch_size*(i+1)]
            idxs = np.random.permutation(y.shape[0])
            x = x[idxs]
            y = y[idxs]

    def add_units_to_conv2d(self, conv2, units):
        dim1 = int(conv2.shape[1])
        dim2 = int(conv2.shape[2])
        dimc = int(units.shape[1])
        repeat_n = dim1*dim2
        units_repeat = RepeatVector(repeat_n)(self.lbl)
        units_repeat = Reshape((dim1, dim2, dimc))(units_repeat)
        return concatenate([conv2, units_repeat])

    def init_model(self, num_classes, shape):
        self.init_session()
        self.init_vectors(num_classes, shape)

        with tf.variable_scope('generator'):
            x = concatenate([self.z, self.lbl])
            x = Dense(7 * 7 * 64, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
            x = Reshape((7, 7, 64))(x)
            x = UpSampling2D(size=(2, 2))(x)

            x = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(x)
            x = Dropout(self.dropout_rate)(x)

            x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = Dropout(self.dropout_rate)(x)
            x = UpSampling2D(size=(2, 2))(x)

            generated = Conv2D(1, kernel_size=(5, 5), activation='sigmoid', padding='same')(x)

        generator = Model([self.z, self.lbl], generated, name='generator')

        with tf.variable_scope('discrim'):
            x = Conv2D(128, kernel_size=(7, 7), strides=(2, 2), padding='same')(self.img)
            x = self.add_units_to_conv2d(x, self.lbl)
            x = LeakyReLU()(x)
            x = Dropout(self.dropout_rate)(x)
            x = MaxPool2D((2, 2), padding='same')(x)

            l = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
            x = LeakyReLU()(l)
            x = Dropout(self.dropout_rate)(x)

            h = Flatten()(x)
            d = Dense(1, activation='sigmoid')(h)

        discrim = Model([self.img, self.lbl], d, name='Discriminator')

        self.generated_z = generator([self.z, self.lbl])

        discr_img = discrim([self.img, self.lbl])
        discr_gen_z = discrim([self.generated_z, self.lbl])

        gan_model = Model([self.z, self.lbl], discr_gen_z, name='GAN')
        self.gan = gan_model([self.z, self.lbl])

        log_dis_img = tf.reduce_mean(-tf.log(discr_img + 1e-10))
        log_dis_gen_z = tf.reduce_mean(-tf.log(1. - discr_gen_z + 1e-10))

        self.L_gen = -log_dis_gen_z
        self.L_dis = 0.5 * (log_dis_gen_z + log_dis_img)

        optimizer_gen = tf.train.RMSPropOptimizer(0.0003)
        optimizer_dis = tf.train.RMSPropOptimizer(0.0001)

        # Переменные генератора и дискриминаторы (отдельно) для оптимизаторов
        generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discrim")

        self.step_gen = optimizer_gen.minimize(self.L_gen, var_list=generator_vars)
        self.step_dis = optimizer_dis.minimize(self.L_dis, var_list=discrim_vars)

        self.sess.run(tf.global_variables_initializer())

    # generator step
    def step(self, image, label, zp):
        l_dis, _ = self.sess.run([self.L_dis, self.step_gen], feed_dict={
            self.z: zp,
            self.lbl: label,
            self.img: image,
            keras_backend.learning_phase(): 1
        })
        return l_dis

    # discriminator step
    def step_d(self, image, label, zp):
        l_dis, _ = self.sess.run([self.L_dis, self.step_dis], feed_dict={
            self.z: zp,
            self.lbl: label,
            self.img: image,
            keras_backend.learning_phase(): 1
        })
        return l_dis

    @staticmethod
    def get_learning_phase():
        return keras_backend.learning_phase()

    def train(self, dataset: Dataset, batch_size=256, epochs=5000):
        l_d = -1

        self.train_batches_it = GANModel.gen_batch(dataset.x_train, dataset.y_train_cat, batch_size)
        self.test_batches_it = GANModel.gen_batch(dataset.x_test, dataset.y_test_cat, batch_size)

        for i in range(epochs):
            print('.', end='')
            b0, b1 = next(self.train_batches_it)
            zp = np.random.randn(batch_size, self.latent_dim)
            # Шаги обучения дискриминатора
            # Достанем новый батч
            for j in range(self.k_step):
                l_d = self.step_d(b0, b1, zp)
                b0, b1 = next(self.train_batches_it)
                zp = np.random.randn(batch_size, self.latent_dim)
                if l_d < 1.0:
                    break
            # Шаги обучения генератора
            for j in range(self.k_step):

                l_d = self.step(b0, b1, zp)
                if l_d > 0.4:
                    break
                b0, b1 = next(self.train_batches_it)
                zp = np.random.randn(batch_size, self.latent_dim)
            # Периодическое рисование результата
            if not i % self.batches_per_period:

                period = i // self.batches_per_period
                for l in self.listeners:
                    l.on_period(period)
                print(l_d)

        for l in self.listeners:
            l.on_finished()

    def add_listener(self, listener: ModelTrainListener):
        self.listeners.append(listener)
