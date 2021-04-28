from loaders.DemoImageLoader import DemoImageLoader
from loaders.InputLoader import InputLoader
from loaders.transforms.AddNoiseTransform import AddNoiseTransform

if __name__ == '__main__':
    trainData, valData = InputLoader.createInputData('./datasets/set5/**/*.png',
                                                    image_loader=DemoImageLoader(),
                                                    method=AddNoiseTransform(noise_amount=0.1))
