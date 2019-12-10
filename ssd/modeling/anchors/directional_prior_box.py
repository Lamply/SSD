from itertools import product

import torch
from math import sqrt


class DirectionalPriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.bin_size = prior_config.BIN_SIZE

    def __call__(self):
        """Generate Directional Prior Boxes.
            It returns the center, height, width and angle of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 5): The prior boxes represented as [[center_x, center_y, w, h, angle]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ang in range(self.bin_size):
                    angle = ang*180.0/self.bin_size
                    priors.append([cx, cy, w, h, angle])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                for ang in range(self.bin_size):
                    angle = ang*180.0/self.bin_size
                    priors.append([cx, cy, w, h, angle])

        priors = torch.tensor(priors)
        return priors
