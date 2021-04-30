import glob
from tensorflow.keras.preprocessing import image
from multiprocessing import Pool
import numpy as np
from sklearn.model_selection import train_test_split

from Config import Config


class InputImageLoader(object):

    def __init__(self, pool=10, test_size=0.2, random_state=32):
        super().__init__()
        self.pool = pool
        self.test_size = test_size
        self.random_state = random_state
    '''
    :param dir_regex: a regex, hogy honnan, miket akarunk betölteni
    '''
    def __call__(self, dir_regex):
        image_files = glob.glob(dir_regex)
        for f in image_files:
            print(f)
        p = Pool(self.pool)
        img_array = p.map(InputImageLoader.read, image_files)
        images_array = np.array(img_array)
        train_x, val_x = train_test_split(images_array, random_state=self.random_state,
                                          test_size=self.test_size)
        return train_x, val_x

    @staticmethod
    def read(path):
        img = image.load_img(path, target_size=Config.image_dim)
        img = image.img_to_array(img)
        img = img / 255
        return img

