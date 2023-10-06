import torch


def adjust_intensity(image, intensity=0.3):
    intensity_born = [-intensity, intensity]
    intensity_factor = torch.empty(1).uniform_(*intensity_born).to(image.device)

    output = image + intensity_factor
    return output


def adjust_brightness(image, brightness=0.5):
    brightness_born = [max(1 - brightness, 0.0), 1 + brightness]
    brightness_factor = torch.empty(1).uniform_(*brightness_born).to(image.device)

    output = image.pow(brightness_factor).clamp_(0, 1)
    return output


def adjust_intensity_per_channel(image, intensity=0.3):
    for channel in range(image.shape[0]):
        image[channel] = adjust_intensity(image[channel], intensity)
    return image


def adjust_brightness_per_channel(image, brightness=0.5):
    for channel in range(image.shape[0]):
        image[channel] = adjust_brightness(image[channel], brightness)
    return image


class ColorJitterPerChannel:
    def __init__(self, intensity=0.3, brightness=0.5):
        self.intensity = intensity
        self.brightness = brightness

    def __call__(self, img):
        img = adjust_intensity_per_channel(img, self.intensity)
        img = adjust_brightness_per_channel(img, self.brightness)

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(intensity={self.intensity}, brightness={self.brightness})"
