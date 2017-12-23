# from dataset.mnist import MnistDataset
from dataset.cifar10 import Cifar10Dataset
from image_saver import ImageSaver
from model import GANModel


def main():
    print('Initializing dataset...')
    data = Cifar10Dataset()
    print('Initializing objects...')
    gan = GANModel(
        latent_dim=2,
        dropout_rate=0.3,
        k_step=5,
        batches_per_period=20,
        dirname='./weights/cifar10_3/'
    )
    saver = ImageSaver(
        gan,
        data.num_classes,
        data.shape,
        n=5,
        gif_filename='./gif/cifar10_3/{}.gif',
        im_filename='./img/cifar10_3/{}.jpg'
    )

    print('Initializing model...')
    gan.init_model(data.num_classes, data.shape, print_summary=True)
    print('Start training model...')
    gan.train(
        data,
        batch_size=256,
        epochs=200
    )
    # gan.save_weights("./weights/cifar10_3.h5")

main()
