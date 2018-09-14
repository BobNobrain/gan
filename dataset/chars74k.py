import os
from dataset.fs_dataset import FsDataset, AugmentationSettings

data_dir = os.environ['CHARS74K_DATA_PATH']
train_dirname = 'train'
test_dirname = 'test'


class Chars74K(FsDataset):
    def __init__(self, batch_size):
        super(Chars74K, self).__init__(
            data_dir,
            (32, 32, 3),
            train_subdir=train_dirname,
            test_subdir=test_dirname,
            train_augmentation=AugmentationSettings(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            ),
            batch_size=batch_size
        )
        if 10000 % batch_size != 0:
            print('[WARNING] Wrong batch size: {} cannot be looped into 1200 images per class!'.format(batch_size))
        # if self.num_classes != 5:
        #     print('[WARNING] Wrong num classes! Expected 5, got {}'.format(self.num_classes))
            # self.num_classes = 5
