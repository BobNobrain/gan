import numpy as np
from scipy.stats import norm
from IPython.display import clear_output

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

from model import GANModel
from model_train_listener import ModelTrainListener

plt.switch_backend('agg')


class ImageSaver(ModelTrainListener):
    def __init__(
            self,
            model: GANModel,
            num_classes,
            shape,
            n=10,
            gif_filename='./gif/manifold_{}.gif',
            gif_title='Label: {}\nBatch: {}',
            im_filename='./img/result_{}.jpg',
            fig_size = (None, None)
    ):
        self.num_classes = num_classes
        self.figs = [[] for _ in range(num_classes)]
        self.periods = []
        self.save_periods = list(range(100)) +\
            list(range(100, 1000, 10)) +\
            list(range(1000, 10000, 500))
        self.n = n
        self.shape = shape
        self.model = model

        fig_w, fig_h = fig_size
        i_w, i_h, _ = shape
        if not fig_w:
            fig_w = (i_w // 28) * n
        if not fig_h:
            fig_h = (i_h // 28) * n
        self.fig_size = (fig_w, fig_h)
        print('Fig size: ')
        print(self.fig_size)

        self.gif_filename = gif_filename
        self.gif_title = gif_title
        self.im_filename = im_filename

        # Так как сэмплируем из N(0, I), то сетку узлов, в которых генерируем цифры,
        # берем из обратной функции распределения
        self.grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        self.grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        self.batches_per_period = 20  # Как часто сохранять картинки

        model.add_listener(self)

    def draw_manifold(self, label, show=True):
        # Рисование цифр из многообразия
        w, h, ch = self.shape
        figure = np.zeros((w * self.n, h * self.n, ch))
        input_lbl = np.zeros((1, self.num_classes))
        input_lbl[0, label] = 1.
        for i, yi in enumerate(self.grid_x):
            for j, xi in enumerate(self.grid_y):
                z_sample = np.zeros((1, self.model.latent_dim))
                z_sample[:, :2] = np.array([[xi, yi]])

                x_generated = self.model.sess.run(
                    self.model.generated_z,
                    feed_dict={
                        self.model.z: z_sample,
                        self.model.lbl: input_lbl,
                        GANModel.get_learning_phase(): 0
                    })
                digit = x_generated[0].squeeze()
                figure[
                    i * w: (i + 1) * w,
                    j * h: (j + 1) * h
                ] = digit
        if show:
            # Визуализация
            plt.figure(figsize=self.fig_size)
            plt.imshow(figure, cmap='Greys')
            plt.grid(False)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show()
        return figure

    def make_2d_figs_gif(self, figs, periods, c, fname, fig, batches_per_period):
        w, h, ch = self.shape
        norm_colors = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
        if ch == 1:
            im = plt.imshow(np.zeros((w, h)), cmap='Greys', norm=norm_colors)
        else:
            im = plt.imshow(np.zeros(self.shape), norm=norm_colors)
        plt.grid(None)
        plt.title(self.gif_title.format(c, 0))

        def update(i):
            im.set_array(figs[i])
            im.axes.set_title(self.gif_title.format(c, periods[i] * batches_per_period))
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)
            return im

        anim = FuncAnimation(fig, update, frames=range(len(figs)), interval=100)
        anim.save(fname, dpi=80, writer='imagemagick')

    def make_result_image(self, figs, c, fname, fig, batches_per_period):
        w, h, ch = self.shape
        norm_colors = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
        if ch == 1:
            im = plt.imshow(np.zeros((w, h)), cmap='Greys', norm=norm_colors)
        else:
            im = plt.imshow(np.zeros(self.shape), norm=norm_colors)

        plt.grid(None)
        plt.title(self.gif_title.format(c, 0))
        im.set_array(figs[len(figs) - 1])
        im.axes.set_title(self.gif_title.format(c, batches_per_period))
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        fig.savefig(fname)

    def on_period(self, period):
        if period in self.save_periods:
            clear_output()  # Не захламляем output
            # Рисование многообразия для рандомного y
            draw_lbl = np.random.randint(0, self.num_classes)
            # print(draw_lbl)
            for label in range(self.num_classes):
                self.figs[label].append(self.draw_manifold(label, show=False))

            self.periods.append(period)

    def on_finished(self):
        for label in range(self.num_classes):
            print('Drawing images for {}...'.format(label))
            try:
                self.make_2d_figs_gif(
                    self.figs[label],
                    self.periods,
                    label,
                    self.gif_filename.format(label),
                    plt.figure(figsize=self.fig_size),
                    self.model.batches_per_period
                )
                print('gif done')
                self.make_result_image(
                    self.figs[label],
                    # self.periods,
                    label,
                    self.im_filename.format(label),
                    plt.figure(figsize=self.fig_size),
                    self.model.batches_per_period
                )
                print('result done')
            except Exception:
                print('Exception catched!')
