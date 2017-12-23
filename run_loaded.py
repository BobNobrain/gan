# from dataset.mnist import MnistDataset
# from dataset.cifar10 import Cifar10Dataset
from image_saver import ImageSaver
from model import GANModel


def main():
    num_classes = 10
    shape = (32, 32, 3)  # cifar10
    gan = GANModel(
        latent_dim=2,
        dropout_rate=0.3,
        k_step=5,
        batches_per_period=20
    )
    saver = ImageSaver(
        gan,
        num_classes,
        shape,
        n=5,
        gif_filename='./gif/cifar10_3/{}.gif',
        im_filename='./img/cifar10_3/{}.jpg'
    )

    print('Initializing model...')
    gan.init_model(num_classes, shape)
    print('Loading weights...')
    gan.load_weights("./weights/cifar10_3.h5")

    print('Drawing images...')
    saver.on_finished(make_gif=False)

main()
