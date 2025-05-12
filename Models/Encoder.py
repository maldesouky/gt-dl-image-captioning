from torch import nn as nn
# from torchvision.models import resnet50
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        cnn_model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        for param in cnn_model.parameters():
            param.requires_grad_(False)

        self.feature_extractor = nn.Sequential(*list(cnn_model.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        features = self.feature_extractor(images)
        features = self.adaptive_pool(features)
        return features.permute(0, 2, 3, 1)  # Batch x H x W x D

