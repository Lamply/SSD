from ssd.modeling.anchors.prior_box import PriorBox
from ssd.modeling.anchors.directional_prior_box import DirectionalPriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if cfg.MODEL.BOX_HEAD.NAME == "SSDRotateBoxHead":
        if is_train:
            transform = [
                ConvertFromInts(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomMirror(),
                ToPercentCoordsAngle(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),
            ]
        else:
            transform = [
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor()
            ]
    else:
        if is_train:
            transform = [
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),
            ]
        else:
            transform = [
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor()
            ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    if cfg.MODEL.BOX_HEAD.NAME == "SSDRotateBoxHead":
        transform = SSDTargetTransform(DirectionalPriorBox(cfg)(),
                                        cfg.MODEL.CENTER_VARIANCE,
                                        cfg.MODEL.SIZE_VARIANCE,
                                        cfg.MODEL.THRESHOLD)
    else:
        transform = SSDTargetTransform(PriorBox(cfg)(),
                                        cfg.MODEL.CENTER_VARIANCE,
                                        cfg.MODEL.SIZE_VARIANCE,
                                        cfg.MODEL.THRESHOLD)
    return transform
