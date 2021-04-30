from loaders.InputImageLoader import InputImageLoader
from loaders.InputTransform import InputTransform
from loaders.TransformMethod import TransformMethod


class DataAccessService(object):

    inputImageLoader = InputImageLoader()
    inputTransform = InputTransform()

    @staticmethod
    def read_images(dir_regex):
        return DataAccessService.inputImageLoader(dir_regex)

    @staticmethod
    def transform_images(train, val, transform_method: TransformMethod):
        return DataAccessService.inputTransform.createInputData(train, val, transform_method)
