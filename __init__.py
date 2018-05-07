# from dataset.mnist import MnistDataset
# from dataset.cifar10 import Cifar10Dataset
# from dataset.linnaeus5_32 import Linnaeus5x32Dataset
from dataset.linnaeus5_64 import Linnaeus5x64Dataset
from image_saver import ImageSaver
# from models.gan import GANModel
from models.dcgan import DCGANModel


def main(
        resume=False,
        epochs=5000,
        print_model_summary=False
):
    print('Initializing dataset...')
    data = Linnaeus5x64Dataset(batch_size=200)
    # data = Cifar10Dataset(batch_size=256)
    print('Initializing objects...')
    # gan = GANModel(
    #     latent_dim=4,
    #     dropout_rate=0.3,
    #     k_step=5,
    #     batches_per_period=20,
    #     dirname='./weights/linnaeus5x32/',
    #     save_period=500
    # )
    model = DCGANModel(
        latent_dim=100,
        batches_per_period=20,
        dirname='./weights/linnaeus5x64-dc/',
        save_period=1000
    )
    saver = ImageSaver(
        model,
        data.num_classes,
        data.shape,
        n=1,
        gif_filename='./gif/lin64-dc/57000/{}.gif',
        im_filename='./img/lin64-dc/57000/{}.jpg'
    )

    print('Initializing model...')
    model.init_model(data.num_classes, data.shape, print_summary=print_model_summary)
    if resume:
        print('Loading weights...')
        model.load_weights()
    print('Start training model...')
    model.train(
        data,
        epochs=epochs
    )


main(
    print_model_summary=True,
    epochs=5000,
    resume=True
)
