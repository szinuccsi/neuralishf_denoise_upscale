import glob
from tensorflow.keras.preprocessing import image
from multiprocessing import Pool
import numpy as np
from sklearn.model_selection import train_test_split

from Config import Config


class DemoImageLoader(object):

    '''
    :param dir_regex: a regex, hogy honnan, miket akarunk betölteni
    '''
    @staticmethod
    def load_images(dir_regex, pool=10):
        image_files = glob.glob(dir_regex)
        for f in image_files:
            print(f)
        p = Pool(pool)
        img_array = p.map(DemoImageLoader.read, image_files)
        images_array = np.array(img_array)
        train_x, val_x = train_test_split(images_array, random_state=32, test_size=0.2)
        return train_x, val_x

    @staticmethod
    def read(path):
        img = image.load_img(path, target_size=Config.image_dim)
        img = image.img_to_array(img)
        img = img / 255
        return img

