
import torchvision
import random
from PIL import Image
import numpy as np
import numbers
import torch

class NumpyToPilImage(object):
    def __call__(self, image):
        return Image.fromarray(image.astype('uint8'))

class GroupNumpyToPILImage(object):
    def __init__(self, use_inverse):
        self.use_inverse = use_inverse

    def __call__(self, Img):
        ImgGroup = []

        for k in range(Img.shape[0]):
            if self.use_inverse:
                kk = Img.shape[0] - k - 1
            else:
                kk = k
            img = Image.fromarray(Img[kk, :, :, :].astype('uint8')).convert('RGB')
            ImgGroup.append(img)

        return ImgGroup


class GroupRandomCrop(object):
    def __init__(self, img_size, crop_size):
        if isinstance(img_size, numbers.Number):
            self.img_size = (int(img_size), int(img_size))
        else:
            self.img_size = img_size

        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, img_group):

        w, h = self.img_size
        th, tw = self.crop_size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self):
        pass

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, Imgs):
        L, C, H, W = Imgs.shape

        tensor = []
        for k in range(L):
            img = self.normalize(Imgs[k, :, :, :])
            tensor.append(img)

        return torch.stack(tensor, dim=0)


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            # return np.concatenate(img_group, axis=2)
            return np.stack(img_group, axis=0)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(0, 3, 1, 2).contiguous()

        return img.float().div(255) if self.div else img.float()

