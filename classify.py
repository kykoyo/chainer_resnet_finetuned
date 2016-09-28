import argparse
import os
import os.path as osp
import glob
import time
import cPickle as pickle
import random

import numpy as np
import cv2
import chainer
from chainer import cuda
import chainer.serializers as S
from chainer import Variable

import util
from classifier import Classifier
from archs import alex, googlenet, googlenetbn, nin

def main():
    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='alex',
                        help='Convnet architecture')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--img_files', nargs='+', required=True)
    parser.add_argument('--label_file', default='labels.txt')
    parser.add_argument('--mean', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
    args = parser.parse_args()

    gpu = args.gpu
    model = archs[args.arch]()
    initmodel = args.initmodel
    insize = model.insize
    cropwidth = 256 - insize
    mean_image = np.load(args.mean)
    img_files = args.img_files
    print('cropwidth',cropwidth)
    
    classifier = Classifier(gpu=gpu, model=model, initmodel=initmodel)
    normalize = False if classifier._use_caffemodel else True
    categories = np.loadtxt(args.label_file, str, delimiter="\t")
    for img_file in img_files:
        print('classify', img_file)
        img = util.load_image(path=img_file, crop_size=model.insize, normalize=normalize, mean_image=mean_image)
        x_batch = np.ndarray(
                (1, 3, insize,insize), dtype=np.float32)
        x_batch[0]=img

        if args.gpu >= 0:
          x_batch=cuda.to_gpu(x_batch)
        x = chainer.Variable(x_batch, volatile=True)
        score = classifier.predict(x)
        top_k = 10
        prediction = zip(score.data[0].tolist(), categories)
        prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
        for rank, (score, name) in enumerate(prediction[:top_k], start=1):
            print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
    
if __name__ == '__main__':
    main()
