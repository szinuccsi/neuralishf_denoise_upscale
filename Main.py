from Config import Config
from loaders.DataAccessService import DataAccessService
from loaders.transforms.AddNoiseTransform import AddNoiseTransform
from net_modules.full_nets.AutoEncoderNetBuilder import AutoEncoderNetBuilder
from net_modules.full_nets.ReferenceNetBuilder import ReferenceNetBuilder
from net_modules.metrics.Metrics import Metrics

model = ReferenceNetBuilder.create_model()
model.compile(optimizer='adam', loss='mse', metrics=[Metrics.PSNR])
model.summary()

model = AutoEncoderNetBuilder.create_model(inputShape=Config.image_dim)
model.compile(optimizer='adam', loss='mse', metrics=[Metrics.PSNR])
model.summary()


if __name__ == '__main__':
    train_y, val_y = DataAccessService.read_images('datasets/set5/**/*.png')
    train_x, val_x = DataAccessService.transform_images(train_y, val_y, AddNoiseTransform(noise_amount=0.1))




