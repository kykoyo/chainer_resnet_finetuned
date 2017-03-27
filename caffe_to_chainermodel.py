import os
import os.path as osp
import pickle

from chainer.functions.caffe import CaffeFunction
import chainer.serializers as s

import util

model_dir = 'pretrained_models'
util.check_dirs(model_dir)
print("Reading caffe model...")
func = CaffeFunction(osp.join(model_dir,"ResNet-101-model.caffemodel")) # read caffe model

print("Writing as chainer model with hdf5...")
s.save_hdf5(osp.join(model_dir,"resnet.h5"), func) # write as chainer model
print("Writing as chainer model with npz...")
s.save_npz(osp.join(model_dir,"resnet.npz"), func) # write as chainer model
print("Writing as chainer model with pickle...")
save_path = osp.join(model_dir, "resnet.pkl")

with open(save_path, 'wb') as f:
  pickle.dump(func, f, protocol=-1)
print("Done")
