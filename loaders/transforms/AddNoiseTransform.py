import numpy as np
import cv2 as cv2

from loaders.TransformMethod import TransformMethod


class AddNoiseTransform(TransformMethod):

    def __init__(self, noise_amount):
        super().__init__()
        self.noise_amount = noise_amount

    def __call__(self, image):
        gauss = np.random.normal(0, self.noise_amount, image.size)
        gauss = gauss.reshape(image.shape).astype('float32')
        return cv2.add(image, gauss)