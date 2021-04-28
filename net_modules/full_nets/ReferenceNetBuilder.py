import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class ReferenceNetBuilder(object):

    @staticmethod
    def create_model():
        Input_img = Input(shape=(None, None, 3))

        # encoding architecture
        x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(
            Input_img)
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
        x3 = MaxPool2D(padding='same')(x2)

        x4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
        x5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
        x6 = MaxPool2D(padding='same')(x5)

        encoded = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)

        # decoding architecture
        x7 = Conv2DTranspose(512, (3, 3), padding='same', strides=2, activation='relu')(encoded)
        # x7 = UpSampling2D()(encoded)
        x8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
        x9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
        x10 = Add()([x5, x9])

        x11 = UpSampling2D()(x10)
        x12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
        x13 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
        x14 = Add()([x2, x13])

        decoded = Conv2D(3, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)

        autoencoder = Model(Input_img, decoded)
        return autoencoder

