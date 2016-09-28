import os
import cPickle as pickle

from chainer.functions.caffe import CaffeFunction
import chainer.serializers as s

if not os.path.exists('pretrained_models'):
    os.makedirs('pretrained_models')
print("Reading caffe model...")
func = CaffeFunction("pretrained_models/bvlc_alexnet.caffemodel") # read caffe model
print("Writing as chainer model with hdf5...")
s.save_hdf5("pretrained_models/alexnet.h5", func) # write as chainer model
print("Writing as chainer model with pickle...")
pickle.dump(func, open('pretrained_models/alexnet.pkl', 'wb'), -1)
print("Done")
