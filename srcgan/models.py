# -*- coding: utf-8 -*-
import functools
import chainer
import chainer.links.caffe
import math
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear
from chainer.links.normalization.batch_normalization import BatchNormalization
from chainer.initializers import Normal
from chainer.functions import depth2space

def pixel_shuffle_upscale(x: chainer.Variable):
    def channel_to_axis(x: chainer.Variable, axis):
        channels = chainer.functions.separate(x, axis=1)
        result_channel = int(len(channels) / 2)
        w1, w2 = chainer.functions.stack(channels[:result_channel], axis=1), chainer.functions.stack(
            channels[result_channel:], axis=1)
        odds, evens = chainer.functions.separate(w1, axis=axis), chainer.functions.separate(w2, axis=axis)
        width_widened = chainer.functions.stack(
            functools.reduce(lambda x, y: x + y, ([a, b] for a, b in zip(odds, evens)))
            , axis=axis)
        return width_widened

    return channel_to_axis(channel_to_axis(x, 2), 3)


class SRGeneratorResBlock(link.Chain):
    def __init__(self):
        super().__init__()

        with self.init_scope():
            self.c1=Convolution2D(64, 64, ksize=3, stride=1, pad=1, initialW=Normal(0.02))
            self.bn1=BatchNormalization(64)
            self.c2=Convolution2D(64, 64, ksize=3, stride=1, pad=1, initialW=Normal(0.02))
            self.bn2=BatchNormalization(64)

    def __call__(self, x: chainer.Variable):
        h = chainer.functions.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return h + x  # residual


class SRGeneratorUpScaleBlock(chainer.Chain):
    def __init__(self):
        super().__init__()

        with self.init_scope():
            self.conv = Convolution2D(in_channels=64, out_channels=256, ksize=3, stride=1, pad=1,
                                      initialW=Normal(0.02))

    def __call__(self, x: chainer.Variable):
        h = self.conv(x)
        h = depth2space(h, r=2)
        h = chainer.functions.relu(h)
        return h


class SRGenerator(chainer.Chain):
    def __init__(self):
        super().__init__()
        
        with self.init_scope():
            self.first=Convolution2D(3, 64, ksize=3, stride=1, pad=1, initialW=Normal(0.02))
            self.res1=SRGeneratorResBlock()
            self.res2=SRGeneratorResBlock()
            self.res3=SRGeneratorResBlock()
            self.res4=SRGeneratorResBlock()
            self.res5=SRGeneratorResBlock()
            self.conv_mid=Convolution2D(64, 64, ksize=3, stride=1, pad=1, initialW=Normal(0.02))
            self.bn_mid=BatchNormalization(64)
            self.upscale1=SRGeneratorUpScaleBlock()
            self.upscale2=SRGeneratorUpScaleBlock()
            self.conv_output=Convolution2D(64, 3, ksize=3, stride=1, pad=1,
                                                    initialW=Normal(0.02))

    def __call__(self, x: chainer.Variable):
        h = first = chainer.functions.relu(self.first(x))

        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        mid = self.bn_mid(self.conv_mid(h))

        h = first + mid

        h = self.upscale2(self.upscale1(h))

        h = self.conv_output(h)
        return h


class SRDiscriminator(chainer.Chain):
    def __init__(self):
        super().__init__()
        
        with self.init_scope():
            self.conv_input=Convolution2D(3, 64, ksize=3, stride=1, pad=0, initialW=Normal(0.02))
            self.c1=Convolution2D(64, 64, ksize=3, stride=2, pad=0, initialW=Normal(0.02))
            self.bn1=BatchNormalization(64)
            self.c2=Convolution2D(64, 128, ksize=3, stride=1, pad=0, initialW=Normal(0.02))
            self.bn2=BatchNormalization(128)
            self.c3=Convolution2D(128, 128, ksize=3, stride=2, pad=0, initialW=Normal(0.02))
            self.bn3=BatchNormalization(128)
            self.c4=Convolution2D(128, 256, ksize=3, stride=1, pad=0, initialW=Normal(0.02))
            self.bn4=BatchNormalization(256)
            self.c5=Convolution2D(256, 256, ksize=3, stride=2, pad=0, initialW=Normal(0.02))
            self.bn5=BatchNormalization(256)
            self.c6=Convolution2D(256, 512, ksize=3, stride=1, pad=0, initialW=Normal(0.02))
            self.bn6=BatchNormalization(512)
            self.c7=Convolution2D(512, 512, ksize=3, stride=2, pad=0, initialW=Normal(0.02))
            self.bn7=BatchNormalization(512)
            self.linear1=Linear(in_size=4608, out_size=1024)
            self.linear2=Linear(in_size=None, out_size=2)

    def __call__(self, x):
        h = self.conv_input(x)
        h = self.bn1(chainer.functions.elu(self.c1(h)))
        h = self.bn2(chainer.functions.elu(self.c2(h)))
        h = self.bn3(chainer.functions.elu(self.c3(h)))
        h = self.bn4(chainer.functions.elu(self.c4(h)))
        h = self.bn5(chainer.functions.elu(self.c5(h)))
        h = self.bn6(chainer.functions.elu(self.c6(h)))
        h = self.bn7(chainer.functions.elu(self.c7(h)))
        h = chainer.functions.elu(self.linear1(h))
        h = chainer.functions.sigmoid(self.linear2(h))
        return h