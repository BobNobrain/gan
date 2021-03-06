import numpy as np

from keras.layers import Dropout, Reshape, Flatten, RepeatVector
from keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from keras import backend as keras_backend
import tensorflow as tf

from dataset.dataset import Dataset
from models.model import Model as CModel


class GANModel(CModel):
    def __init__(
            self,
            # num_classes,
            latent_dim=2,
            dropout_rate=0.3,
            k_step=5,
            batches_per_period=20,
            dirname='./weights/',
            save_period=100
    ):
        super(GANModel, self).__init__(
            latent_dim=latent_dim,
            batches_per_period=batches_per_period,
            dirname=dirname,
            save_period=save_period
        )
        self.k_step = k_step  # Количество шагов, которые могут делать дискриминатор и генератор во внутреннем цикле
        self.dropout_rate = dropout_rate

        self.L_gen = None
        self.L_dis = None

        self.step_gen = None
        self.step_dis = None

        self.generated_z = None
        self.gan = None
        self.gan_model = None

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

    def add_units_to_conv2d(self, conv2, units):
        dim1 = int(conv2.shape[1])
        dim2 = int(conv2.shape[2])
        dimc = int(units.shape[1])
        repeat_n = dim1*dim2
        units_repeat = RepeatVector(repeat_n)(self.lbl)
        units_repeat = Reshape((dim1, dim2, dimc))(units_repeat)
        return concatenate([conv2, units_repeat])

    def init_model(self, num_classes, shape, print_summary=False):
        self.init_session()
        self.init_vectors(num_classes, shape)
        w, h, channels = shape
        if w % 4 or h % 4:
            raise ValueError(
                'Wrong data shape! Only 4x sizes are supported now! Got {w}x{h}'
                .format(w=w, h=h)
            )
        w = w // 4
        h = h // 4

        with tf.variable_scope('generator'):
            x = concatenate([self.z, self.lbl])
            x = Dense(w * h * 64, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
            x = Reshape((w, h, 64))(x)
            x = UpSampling2D(size=(2, 2))(x)

            x = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(x)
            x = Dropout(self.dropout_rate)(x)

            x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = Dropout(self.dropout_rate)(x)

            x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = Dropout(self.dropout_rate)(x)
            x = UpSampling2D(size=(2, 2))(x)

            generated = Conv2D(channels, kernel_size=(5, 5), activation='sigmoid', padding='same')(x)

        generator = Model([self.z, self.lbl], generated, name='generator')
        if print_summary:
            generator.summary(line_length=120)

        with tf.variable_scope('discrim'):
            x = Conv2D(128, kernel_size=(7, 7), strides=(2, 2), padding='same')(self.img)
            x = self.add_units_to_conv2d(x, self.lbl)
            x = LeakyReLU()(x)
            x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
            x = LeakyReLU()(x)
            x = Dropout(self.dropout_rate)(x)
            x = MaxPool2D((2, 2), padding='same')(x)

            l = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
            x = LeakyReLU()(l)
            x = Dropout(self.dropout_rate)(x)

            h = Flatten()(x)
            d = Dense(1, activation='sigmoid')(h)

        discrim = Model([self.img, self.lbl], d, name='Discriminator')
        if print_summary:
            discrim.summary(line_length=120)

        self.generated_z = generator([self.z, self.lbl])

        discr_img = discrim([self.img, self.lbl])
        discr_gen_z = discrim([self.generated_z, self.lbl])

        gan_model = Model([self.z, self.lbl], discr_gen_z, name='GAN')
        self.gan_model = gan_model
        self.gan = gan_model([self.z, self.lbl])

        log_dis_img = tf.reduce_mean(-tf.log(discr_img + 1e-10))
        log_dis_gen_z = tf.reduce_mean(-tf.log(1. - discr_gen_z + 1e-10))

        self.L_gen = -log_dis_gen_z
        self.L_dis = 0.5 * (log_dis_gen_z + log_dis_img)

        optimizer_gen = tf.train.RMSPropOptimizer(0.0003)
        optimizer_dis = tf.train.RMSPropOptimizer(0.0001)

        # Gen & discr variables (separated) for optimizers
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

    # @staticmethod
    def get_learning_phase():
        return keras_backend.learning_phase()

    def train(self, dataset: Dataset, epochs=5000):
        l_d = -1
        l_g = -1

        train_batches_it = dataset.train_batch_iterator()
        # TODO: why this is unused?
        # test_batches_it = dataset.test_batch_iterator()

        for i in range(epochs):
            # print('.', end='')
            b0, b1 = next(train_batches_it)
            zp = np.random.randn(dataset.batch_size, self.latent_dim)
            # Шаги обучения дискриминатора
            # Достанем новый батч
            for j in range(self.k_step):
                l_d = self.step_d(b0, b1, zp)
                b0, b1 = next(train_batches_it)
                zp = np.random.randn(dataset.batch_size, self.latent_dim)
                if l_d < 1.0:
                    break
            # Шаги обучения генератора
            for j in range(self.k_step):
                l_g = self.step(b0, b1, zp)
                if l_g > 0.4:
                    break
                b0, b1 = next(train_batches_it)
                zp = np.random.randn(dataset.batch_size, self.latent_dim)

            # сохраняем модель каждые save_period эпох
            if i % self.save_period == self.save_period - 1:
                print('Saving model at {}...'.format(i))
                self.save_weights(i)

            # Периодическое рисование результата
            if not i % self.batches_per_period:

                period = i // self.batches_per_period
                for l in self.listeners:
                    l.on_period(period)
                print('epoch: {epoch}; loss_gen: {loss_gen}; loss_dis: {loss_dis}'.format(
                    epoch=i,
                    loss_dis=l_d,
                    loss_gen=l_g
                ))

        for l in self.listeners:
            l.on_finished()

    def feed(self, z_sample, input_lbl):
        return self.sess.run(
            self.generated_z,
            feed_dict={
                self.z: z_sample,
                self.lbl: input_lbl,
                GANModel.get_learning_phase(): 0
            }
        )
