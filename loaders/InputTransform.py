import numpy as np


class InputTransform(object):

    def createInputData(self, train, val, method):
        train_x_px = []

        for i in range(train.shape[0]):
            temp = method(train[i, :, :, :])
            train_x_px.append(temp)

        train_x_px = np.array(train_x_px)

        val_x_px = []

        for i in range(val.shape[0]):
            temp = method(val[i, :, :, :])
            val_x_px.append(temp)

        val_x_px = np.array(val_x_px)

        return train_x_px, val_x_px