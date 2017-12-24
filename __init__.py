from dataset.mnist import MnistDataset
# from dataset.cifar10 import Cifar10Dataset
from image_saver import ImageSaver
from model import GANModel


def main():
    print('Initializing dataset...')
    data = MnistDataset()
    print('Initializing objects...')
    gan = GANModel(
        latent_dim=2,
        dropout_rate=0.3,
        k_step=5,
        batches_per_period=20,
        dirname='./weights/mnist_2/',
        save_period=500
    )
    saver = ImageSaver(
        gan,
        data.num_classes,
        data.shape,
        n=10,
        gif_filename='./gif/mnist_2/{}.gif',
        im_filename='./img/mnist_2/{}.jpg'
    )

    print('Initializing model...')
    gan.init_model(data.num_classes, data.shape, print_summary=True)
    print('Start training model...')
    gan.train(
        data,
        batch_size=256,
        epochs=4000
    )

main()
