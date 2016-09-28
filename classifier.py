import os
import os.path as osp
import cPickle as pickle

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.serializers as S
from chainer import Variable
from chainer.functions import caffe

import util

class Classifier(object):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    """
    def __init__(self, gpu, model, initmodel):
        self.gpu = gpu
        self.model = model
        if 'caffe' in initmodel:
            self._load_caffemodel(initmodel)
            self._use_caffemodel = True
        else:
            self._load_chainermodel(initmodel)
            self._use_caffemodel = False

        if self.gpu != -1:
            self.model.to_gpu(self.gpu)


    def _load_caffemodel(self, initmodel):
        print('Reading caffe model...')
        self.caffemodel = caffe.CaffeFunction(initmodel)
        if self.gpu != -1:
            self.caffemodel.to_gpu(self.gpu)
        
    def _load_chainermodel(self, initmodel):
        print('Reading chainer model...')
        if 'pkl' in initmodel:
            self.initmodel = pickle.load(open(initmodel))
            util.copy_model(self.initmodel, self.model)
        else:
            self.initmodel = initmodel
            S.load_hdf5(initmodel, self.model)

    def predict(self, x):
        if self._use_caffemodel:
            y, = self.caffemodel(inputs={'data': x}, outputs=['fc8'], train=False)
            score = F.softmax(y)
            return score
        else:
            self.model.train = False
            score = self.model.predict(x)
            return score
