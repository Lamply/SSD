from ssd.modeling import registry
from .box_head import SSDBoxHead
from .box_head import SSDRotateBoxHead

__all__ = ['build_box_head', 'SSDBoxHead', 'SSDRotateBoxHead']


def build_box_head(cfg):
    return registry.BOX_HEADS[cfg.MODEL.BOX_HEAD.NAME](cfg)
