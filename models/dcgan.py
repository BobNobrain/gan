import numpy as np

from keras.layers import (Reshape, Input, Convolution2D, Deconvolution2D,
                          BatchNormalization, Activation, Lambda, merge)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from keras import backend as K
import tensorflow as tf

from dataset.dataset import Dataset
from models.model import Model as CModel


def deconv(i, nop, kw, oh, ow, std=1, tail=True, bm='same'):
    i = Deconvolution2D(
        nop, kw, kw,
        subsample=(std, std),
        border_mode=bm,
        output_shape=(None, oh, ow, nop)
    )(i)
    if tail:
        i = BatchNormalization()(i)
        i = LeakyReLU(.2)(i)

    return i


def conv(i, nop, kw, std=1, usebn=True, bm='same'):
    i = Convolution2D(nop, kw, kw, border_mode=bm, subsample=(std, std))(i)
    if usebn:
        i = BatchNormalization()(i)
    i = LeakyReLU(.2)(i)
    return i


def concat_diff(i):  # batch discrimination -  increase generation diversity.
    bv = Lambda(lambda x: K.mean(K.abs(x[:] - K.mean(x, axis=0)), axis=-1, keepdims=True))(i)
    i = merge([i, bv], mode='concat')

    return i


class DCGANModel(CModel):
    def __init__(
            self,
            latent_dim=2,
            batches_per_period=20,
            dirname='./weights/',
            save_period=100
    ):
        super(DCGANModel, self).__init__(
            latent_dim=latent_dim,
            batches_per_period=batches_per_period,
            dirname=dirname,
            save_period=save_period
        )
        self.model = None
        self.gm = None

    def gen(self, shape):  # generative network, 2
        inp = Input(shape=(self.latent_dim,))
        i = inp
        i = Reshape((1, 1, self.latent_dim))(i)

        ngf = 24

        (img_w, img_h, ch) = shape

        if img_w == 32 and img_h == 32:
            i = deconv(i, nop=ngf * 8, kw=4, oh=4, ow=4, std=1, bm='valid')
            i = deconv(i, nop=ngf * 4, kw=4, oh=8, ow=8, std=2)
            i = deconv(i, nop=ngf * 2, kw=4, oh=16, ow=16, std=2)
            i = deconv(i, nop=ngf * 1, kw=4, oh=32, ow=32, std=2)
        elif img_w == 64 and img_h == 64:
            i = deconv(i, nop=ngf * 16, kw=4, oh=4, ow=4, std=1, bm='valid')
            i = deconv(i, nop=ngf * 8, kw=4, oh=8, ow=8, std=2)
            i = deconv(i, nop=ngf * 4, kw=4, oh=16, ow=16, std=2)
            i = deconv(i, nop=ngf * 2, kw=4, oh=32, ow=32, std=2)
            i = deconv(i, nop=ngf * 1, kw=4, oh=64, ow=64, std=2)
        else:
            raise ValueError('Expect 32x32 or 64x64 dataset')

        i = deconv(i, nop=3, kw=4, oh=img_h, ow=img_w, std=1, tail=False)
        i = Activation('tanh')(i)

        m = Model(input=inp, output=i)

        return m

    def dis(self, shape):  # discriminative network, 2
        inp = Input(shape=shape)
        i = inp

        ndf = 24

        i = conv(i, ndf * 1, 4, std=2, usebn=False)
        i = concat_diff(i)
        i = conv(i, ndf * 2, 4, std=2)
        i = concat_diff(i)
        i = conv(i, ndf * 4, 4, std=2)
        i = concat_diff(i)
        i = conv(i, ndf * 8, 4, std=2)
        i = concat_diff(i)
        i = conv(i, ndf * 16, 4, std=2)
        i = concat_diff(i)

        # 1x1
        i = Convolution2D(1, 2, 2, border_mode='valid')(i)

        i = Activation('linear', name='conv_exit')(i)
        i = Activation('sigmoid')(i)

        i = Reshape((1,))(i)

        m = Model(input=inp, output=i)
        return m

    def gan(self, g, d):
        # initialize a GAN trainer

        # this is the fastest way to train a GAN in Keras
        # two models are updated simutaneously in one pass

        noise = Input(shape=g.input_shape[1:])
        real_data = Input(shape=d.input_shape[1:])

        generated = g(noise)
        gscore = d(generated)
        rscore = d(real_data)

        def log_eps(i):
            return K.log(i + 1e-11)

        # single side label smoothing: replace 1.0 with 0.9
        dloss = - K.mean(log_eps(1 - gscore) + .1 * log_eps(1 - rscore) + .9 * log_eps(rscore))
        gloss = - K.mean(log_eps(gscore))

        Adam = tf.train.AdamOptimizer

        lr, b1 = 1e-4, .2  # otherwise won't converge.
        optimizer = Adam(lr, beta1=b1)

        grad_loss_wd = optimizer.compute_gradients(dloss, d.trainable_weights)
        update_wd = optimizer.apply_gradients(grad_loss_wd)

        grad_loss_wg = optimizer.compute_gradients(gloss, g.trainable_weights)
        update_wg = optimizer.apply_gradients(grad_loss_wg)

        def get_internal_updates(model):
            # get all internal update ops (like moving averages) of a model
            inbound_nodes = model._inbound_nodes
            input_tensors = []
            for ibn in inbound_nodes:
                input_tensors += ibn.input_tensors
            updates = [model.get_updates_for(i) for i in input_tensors]
            return updates

        other_parameter_updates = [get_internal_updates(m) for m in [d, g]]
        # those updates includes batch norm.

        # print('other_parameter_updates for the models(mainly for batch norm):')
        # print(other_parameter_updates)

        train_step = [update_wd, update_wg, other_parameter_updates]
        losses = [dloss, gloss]

        learning_phase = K.learning_phase()

        def gan_feed(sess, batch_image, z_input):
            # actual GAN trainer
            nonlocal train_step, losses, noise, real_data, learning_phase

            res = sess.run([train_step, losses], feed_dict={
                noise: z_input,
                real_data: batch_image,
                learning_phase: True
            })

            loss_values = res[1]
            return loss_values  # [dloss,gloss]

        return gan_feed

    def init_model(self, num_classes, shape, print_summary=False):
        gm = self.gen(shape=shape)
        if print_summary:
            gm.summary(line_length=120)

        dm = self.dis(shape=shape)
        if print_summary:
            dm.summary(line_length=120)

        self.model = self.gan(g=gm, d=dm)
        self.gm = gm
        self.sess = K.get_session()

    def train(self, dataset: Dataset, epochs=5000):
        sess = K.get_session()
        if sess != self.sess:
            raise ValueError('Sessions do not match!')
        train_batches_it = dataset.train_batch_iterator()

        for i in range(epochs):
            z_input = np.random.normal(loc=0., scale=1., size=(dataset.batch_size, self.latent_dim))
            images, labels = next(train_batches_it)

            losses = self.model(sess, images, z_input)
            l_d = losses[0]
            l_g = losses[1]

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
        # ignoring input_lbl for now, as model is unconditional
        i = np.random.normal(loc=0., scale=1., size=(1, self.latent_dim))
        gened = self.gm.predict(i, batch_size=1)
        min_pix = np.amin(gened)
        d_pix = np.amax(gened) - min_pix
        gened -= min_pix
        gened *= 1. / d_pix

        return gened

