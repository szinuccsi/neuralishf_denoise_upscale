import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class Metrics(object):

    @staticmethod
    def PSNR(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)