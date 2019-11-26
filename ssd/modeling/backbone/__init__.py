from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .resnet18 import ResNet18_SSD

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'ResNet18_SSD']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
