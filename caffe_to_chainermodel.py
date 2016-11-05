import os
import os.path as osp
import pickle

from chainer.functions.caffe import CaffeFunction
import chainer.serializers as s

import util

model_dir = 'pretrained_models'
util.check_dirs(model_dir)
print("Reading caffe model...")
func = CaffeFunction(osp.join(model_dir,"bvlc_alexnet.caffemodel")) # read caffe model
print("Writing as chainer model with hdf5...")
s.save_hdf5(osp.join(model_dir,"alexnet.h5"), func) # write as chainer model
print("Writing as chainer model with pickle...")
save_path = osp.join(model_dir, "alexnet.plt")
with open(save_path, 'wb') as f:
  pickle.dump(func, f, protocol=-1)
print("Done")
