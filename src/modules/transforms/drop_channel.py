import torch


class DropTransform:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        drop = torch.rand(1) < self.p

        if drop:
            if img.ndim == 3:  # C x H x W
                channel_to_drop = torch.randint(0, img.shape[0], (1,))
                img[channel_to_drop] = 0.0
            elif img.ndim == 4:  # B x C x H x W
                channel_to_drop = torch.randint(0, img.shape[1], (1,))
                img[:, channel_to_drop] = 0.0

        return img
