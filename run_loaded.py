# from dataset.mnist import MnistDataset
# from dataset.cifar10 import Cifar10Dataset
from image_saver import ImageSaver
# from models.gan import GANModel
from models.dcgan import DCGANModel


def main():
    num_classes = 10
    shape = (32, 32, 3)  # cifar10
    # shape = (28, 28, 1)  # mnist
    model = DCGANModel(
        latent_dim=100,
        batches_per_period=20,
        dirname='./weights/linnaeus5x32-dc/',
        save_period=500
    )
    saver = ImageSaver(
        model,
        num_classes,
        shape,
        n=5,
        im_filename='./img/lin32-dc/loaded_{}.jpg'
    )

    print('Initializing model...')
    model.init_model(num_classes, shape)
    model.load_weights()

    print('Drawing images...')
    saver.on_period(saver.save_periods[0])
    saver.on_finished(make_gif=False)


main()
