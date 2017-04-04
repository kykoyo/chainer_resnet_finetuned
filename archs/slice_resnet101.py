#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import chainer
import chainer.functions as F
import chainer.links as L


class BottleNeckA(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(out_size),

            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True),
            bn4=L.BatchNormalization(out_size),
        )

    def __call__(self, x, train):
        h1 = F.relu(self.bn1(self.conv1(x), test=not train))
        h1 = F.relu(self.bn2(self.conv2(h1), test=not train))
        h1 = self.bn3(self.conv3(h1), test=not train)
        h2 = self.bn4(self.conv4(x), test=not train)

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):
    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(in_size),
        )

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = F.relu(self.bn2(self.conv2(h), test=not train))
        h = self.bn3(self.conv3(h), test=not train)

        return F.relu(h + x)


class Block(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB(out_size, ch))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)

        return x


class SliceNet(chainer.Chain):
    """Slice branch network

    see https://arxiv.org/abs/1612.06543
    """
    def __init__(self, slice_axis='h', conv_outch=320, shortside_conv_ksize=5, longside_pooling_ksize=5, pooling_stride=3):
        if slice_axis == 'h':
            ksize = (shortside_conv_ksize, 224)
        elif slice_axis == 'v':
            ksize = (224, shortside_conv_ksize)
        super(SliceNet, self).__init__(
            conv1_1 = L.Convolution2D(3, conv_outch, ksize=ksize, stride=1, pad=0),
            bn1_1 = L.BatchNormalization(conv_outch)
        )
        self.train = True
        self.slice_axis = slice_axis
        self.longside_pooling_ksize = longside_pooling_ksize
        self.pooling_stride = pooling_stride

    def __call__(self, x):
        if self.slice_axis == 'h':
            ksize=(self.longside_pooling_ksize, 1)
        elif self.slice_axis == 'v':
            ksize=(1, self.longside_pooling_ksize)
        h = F.relu(self.bn1_1(self.conv1_1(x), test=not self.train), use_cudnn=False)
        h = F.max_pooling_2d(h, ksize=ksize, stride=self.pooling_stride)
        return h


class SliceResNet(chainer.Chain):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        super(SliceResNet, self).__init__(
            slicenet_h = SliceNet(slice_axis='h'),
            #slicenet_v = SliceNet(slice_axis='v')
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 64, 256, 1),
            res3=Block(4, 256, 128, 512),
            res4=Block(23, 512, 256, 1024),
            res5=Block(3, 1024, 512, 2048),
            fc=L.Linear(25408, 25),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        batchsize = x.shape[0]

        # residual branch
        hr = self.bn1(self.conv1(x), test=not self.train)
        hr = F.max_pooling_2d(F.relu(hr), 3, stride=2)
        hr = self.res2(hr, self.train)
        hr = self.res3(hr, self.train)
        hr = self.res4(hr, self.train)
        hr = self.res5(hr, self.train)
        hr = F.average_pooling_2d(hr, 7, stride=1)
        hr = F.reshape(hr, shape=(batchsize, -1))
        
        # Slice branch
        hs_h = self.slicenet_h(x)
        hs_h = F.reshape(hs_h, shape=(batchsize, -1))
        #hs_v = self.slicenet_v(x)
        #hs_v = F.reshape(hs_h, shape=(batchsize, -1))

        # concat
        h = F.concat([hr, hs_h], axis=1) # or  h = F.concat([hr, hs_h, hs_v], axis=1)

        h = self.fc(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

        # if self.train:
        #     self.loss = F.softmax_cross_entropy(h, t)
        #     self.accuracy = F.accuracy(h, t)
        #     return self.loss
        # else:
        #     return h
