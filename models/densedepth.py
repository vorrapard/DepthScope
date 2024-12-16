import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class UpSample(nn.Sequential):
    def __init__(self, skip_input: int, output_features: int) -> None:
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))

class Encoder(nn.Module):
    def __init__(self, pretrained:bool) -> None:
        super(Encoder, self).__init__()
        self.original_model = models.densenet169(pretrained=pretrained)

    def forward(self, x) -> list:
        features = [x]
        for key, value in self.original_model.features._modules.items():
            features.append(value(features[-1]))
        return features

class Decoder(nn.Module):
    def __init__(self, num_features:int=1664, decoder_width:float=1.0) -> None:
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features) -> nn.modules.Conv2d:
        x_block0 = features[3]
        x_block1 = features[4]
        x_block2 = features[6]
        x_block3 = features[8]
        x_block4 = features[12]

        x_d0 = self.conv2(F.relu(x_block4))
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)

        return self.conv3(x_d4)

class DenseDepth(nn.Module):
    def __init__(self, encoder_pretrained:bool=True) -> None:
        super(DenseDepth, self).__init__()
        self.encoder = Encoder(encoder_pretrained)
        self.decoder = Decoder()

    def forward(self, x) -> nn.Module:
        return self.decoder(self.encoder(x))
