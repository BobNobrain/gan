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
        batches_per_period=20
    )
    saver = ImageSaver(gan, 10)

    print('Initializing model...')
    gan.init_model(data.num_classes, data.shape)
    print('Start training model...')
    gan.train(
        data,
        batch_size=256,
        epochs=5000
    )

main()
