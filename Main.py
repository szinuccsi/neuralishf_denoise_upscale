from loaders.DemoImageLoader import DemoImageLoader
from loaders.InputLoader import InputLoader
from loaders.transforms.AddNoiseTransform import AddNoiseTransform
from net_modules.full_nets.ReferenceNetBuilder import ReferenceNetBuilder
from net_modules.metrics.Metrics import Metrics

model = ReferenceNetBuilder.create_model()
model.compile(optimizer='adam', loss='mse', metrics=[Metrics.PSNR])
model.summary()

if __name__ == '__main__':
    trainData, valData = InputLoader.createInputData('./datasets/set5/**/*.png',
                                                    image_loader=DemoImageLoader(),
                                                    method=AddNoiseTransform(noise_amount=0.1))


