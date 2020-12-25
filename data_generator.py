from torch.utils.data.dataset import Dataset
import numpy as np
import utils
import cv2


class DataGenerator(Dataset):
    def __init__(self, hp):
        super(DataGenerator, self).__init__()
        self.hp = hp
        self.train_data_path = hp['train_data_path']
        self.train_images, self.train_labels = utils.load_training_data(self.train_data_path)
        self.total_train_images = self.train_images.shape[0]

    def __getitem__(self, item):
        np.random.seed()

        select_random_img = np.random.randint(self.train_images.shape[0])
        image = self.train_images[select_random_img, :]
        image = image.reshape(3, 32, 32)
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, (224, 224))
        image = np.transpose(image, (2, 0, 1))
        label = self.train_labels[select_random_img]

        return image, label

    def __len__(self):
        return self.total_train_images


class TestDataGenerator(Dataset):
    def __init__(self, test_data_path):
        super(TestDataGenerator, self).__init__()
        self.test_data_path = test_data_path
        self.test_images, self.test_labels = utils.load_test_data(self.test_data_path)
        self.total_test_images = self.test_images.shape[0]

    def __getitem__(self, item):

        image = self.test_images[item, :]
        image = image.reshape(3, 32, 32)
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, (224, 224))
        image = np.transpose(image, (2, 0, 1))
        label = self.test_labels[item]

        return image, label

    def __len__(self):
        return self.total_test_images
