import os
from keras.preprocessing.image import ImageDataGenerator
from dataset.dataset import Dataset


class AugmentationSettings:
    def __init__(
            self,
            rescale=1. / 255,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False
    ):
        self.rescale = rescale
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip


class FsDataset(Dataset):
    def __init__(
            self,
            data_dir,
            shape,
            train_subdir='train',
            test_subdir='test',
            train_augmentation=AugmentationSettings(),
            test_augmentation=AugmentationSettings(),
            batch_size=32
    ):
        super(FsDataset, self).__init__(batch_size=batch_size)
        self.shape = shape

        train_dir_name = os.path.join(data_dir, train_subdir)
        self.num_classes = len([
            name for name in os.listdir(train_dir_name)
            if os.path.isdir(os.path.join(train_dir_name, name))
        ])

        train_datagen = ImageDataGenerator(
            rescale=train_augmentation.rescale,
            shear_range=train_augmentation.shear_range,
            zoom_range=train_augmentation.zoom_range,
            horizontal_flip=train_augmentation.horizontal_flip
        )

        test_datagen = ImageDataGenerator(
            rescale=test_augmentation.rescale,
            shear_range=test_augmentation.shear_range,
            zoom_range=test_augmentation.zoom_range,
            horizontal_flip=test_augmentation.horizontal_flip
        )

        (img_width, img_height, channels) = self.shape
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, train_subdir),
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )

        self.test_generator = test_datagen.flow_from_directory(
            os.path.join(data_dir, test_subdir),
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical'
        )

    def train_batch_iterator(self):
        return self.train_generator

    def test_batch_iterator(self):
        return self.test_generator
