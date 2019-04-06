# -*- coding: utf-8 -*-
import os
import six
import numpy
import math
import cv2
import random
from PIL import Image
from chainer.dataset import dataset_mixin


class PILImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, src, resize=None):
        self._files = os.listdir(src)
        self._resize = resize
        self._src = src

    def __len__(self):
        return len(self._files)

    def get_example(self, i) -> Image:
        path = os.path.join(self._src, self._files[i])
        original_image = Image.open(path)
        if self._resize is not None:
            return original_image.resize(self._resize)
        else:
            return original_image


class ResizedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, src, resize=None, dtype=numpy.float32):
        self.base = PILImageDataset(src=src, resize=resize)
        self._dtype = dtype

    def __len__(self):
        return len(self.base)

    def get_example(self, i) -> numpy.ndarray:
        image = self.base[i]
        image_ary = numpy.asarray(image, dtype=self._dtype)
        image_data = image_ary.transpose(2, 0, 1)
        return image_data


class PreprocessedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, cropsize, src, resize=None, dtype=numpy.float32):
        self.base = ResizedImageDataset(resize=resize, src=src)
        self._dtype = dtype
        self.cropsize = cropsize

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        x = random.randint(0, image.shape[1] - self.cropsize)
        y = random.randint(0, image.shape[2] - self.cropsize)

        cropeed_high_res = image[:, x:x + self.cropsize, y:y + self.cropsize]
        cropped_low_res = cv2.resize(cropeed_high_res.transpose(1, 2, 0), dsize=(int(self.cropsize/4), int(self.cropsize/4)),
                                     interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
        return cropped_low_res, cropeed_high_res
