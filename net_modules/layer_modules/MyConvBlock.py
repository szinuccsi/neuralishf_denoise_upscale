import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class ConvBlock(tf.keras.Model):

    def __init__(self, filters, kernel_size, activation, strides, dilation_rate, kernel_regularizer_param, bNorm,
                 dropout_rate,
                 groups=1, data_format='channels_first'):
        super().__init__()
        self.convLayer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                                                padding='same',
                                                data_format=data_format, dilation_rate=dilation_rate, groups=groups,
                                                activation=activation,
                                                kernel_regularizer=regularizers.l1(kernel_regularizer_param)
                                                )
        if bNorm:
            self.batchNormLayer = tf.keras.layers.BatchNormalization()
        else:
            self.batchNormLayer = tf.identity
        self.dropoutLayer = tf.keras.layers.SpatialDropout2D(
            rate=dropout_rate, data_format=data_format
        )

    def __call__(self, input, training):
        return self.dropoutLayer(self.batchNormLayer(self.convLayer(input)), training)