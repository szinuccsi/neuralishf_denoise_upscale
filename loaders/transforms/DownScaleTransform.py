import cv2 as cv2

class DownScaleTransform(object):

    def __init__(self, downscale_percent):
        super().__init__()
        self.downscale_percent = downscale_percent

    def __call__(self, image):
        orig_dim = (image.shape[1], image.shape[0])
        new_width = int(image.shape[1] * self.downscale_percent / 100)
        new_height = int(image.shape[0] * self.downscale_percent / 100)
        new_dim = (new_width, new_height)

        small_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

        return cv2.resize(small_image, orig_dim, interpolation=cv2.INTER_AREA)
