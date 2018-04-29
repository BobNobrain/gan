# from dataset.mnist import MnistDataset
# from dataset.cifar10 import Cifar10Dataset
from image_saver import ImageSaver
from model import GANModel


def main():
    num_classes = 10
    # shape = (32, 32, 3)  # cifar10
    shape = (28, 28, 1)  # mnist
    gan = GANModel(
        latent_dim=2,
        dropout_rate=0.3,
        k_step=5,
        batches_per_period=20,
        dirname='./weights/mnist_2/'
    )
    saver = ImageSaver(
        gan,
        num_classes,
        shape,
        n=15,
        im_filename='./img/mnist_2/loaded_{}_timing.jpg'
    )

    print('Initializing model...')
    gan.init_model(num_classes, shape)
    gan.load_weights()

    print('Drawing images...')
    saver.on_period(20)
    saver.on_finished(make_gif=False)

main()
