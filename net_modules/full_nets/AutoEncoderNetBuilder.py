import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class AutoEncoderNetBuilder(object):

    @classmethod
    def create_model(cls, inputShape=(200, 200, 3), nFeat=16, nLevel=2, nLayersPerLevel=2, kernel_size=(3, 3),
                          activation='relu', kernel_regularizer_param=1e-7):

        input = Input(shape=inputShape)
        downLevels = []

        filters = nFeat
        downLevels.append(cls.createConvLevel(input,
                                              1,
                                              filters, kernel_size, activation, kernel_regularizer_param))
        for i in range(nLevel + 1):
            if i > 0:
                downLevels.append(MaxPool2D(padding='same')(downLevels[i - 1]))
            filters = 2 * filters
            downLevels[i] = cls.createConvLevel(downLevels[i],
                                                nLayersPerLevel,
                                                filters, kernel_size, activation, kernel_regularizer_param)
        upLevels = [None for i in range(nLevel + 1)]
        upLevels[nLevel] = downLevels[nLevel]
        for i in range(nLevel - 1, -1, -1):
            upLevels[i] = UpSampling2D(size=(2, 2), data_format=cls.NetConfig.data_format, interpolation=cls.NetConfig.upsampling_interpolation)(upLevels[i + 1])
            filters = filters / 2
            upLevels[i] = cls.createConvLevel(upLevels[i],
                                              1,
                                              filters, kernel_size, activation, kernel_regularizer_param)
            upLevels[i] = Add()([upLevels[i], downLevels[i]])
            upLevels[i] = cls.createConvLevel(upLevels[i],
                                              nLayersPerLevel,
                                              filters, kernel_size, activation, kernel_regularizer_param)
        output = cls.createConvLevel(upLevels[0],
                                     inputShape[-1],
                                     filters, kernel_size, activation, kernel_regularizer_param)
        model = Model(input, output)
        return model

    @classmethod
    def createConvLevel(cls, input,
                        nLayersPerLevel,
                        filters, kernel_size, activation, kernel_regularizer_param):
        output = input

        for i in range(nLayersPerLevel):
            output = cls.createConvBlock(output,
                                         filters, kernel_size, activation, kernel_regularizer_param)
        return output

    @classmethod
    def createConvBlock(cls, input,
                        filters, kernel_size, activation, kernel_regularizer_param):
        convLayer = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=cls.NetConfig.stride, padding='same',
            data_format=cls.NetConfig.data_format, dilation_rate=cls.NetConfig.dilation_rate, groups=cls.NetConfig.groups,
            activation=activation,
            kernel_regularizer=regularizers.l1(kernel_regularizer_param)
        )(input)
        dropoutInput = convLayer
        if cls.NetConfig.bNorm:
            dropoutInput = tf.keras.layers.BatchNormalization()(convLayer)
        output = tf.keras.layers.SpatialDropout2D(
            rate=cls.NetConfig.dropout_rate, data_format=cls.NetConfig.data_format
        )(dropoutInput)
        return output

    class NetConfig:
        inputShape = (None, None, 3)
        bNorm = True
        dropout_rate = 0.1
        groups = 1
        data_format = "channels_last"
        stride = (1, 1)
        dilation_rate = (1, 1)
        upsampling_interpolation = 'bilinear'

