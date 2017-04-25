import argparse
import os
import os.path as osp
import glob
import time
import cPickle as pickle
import random

import pandas as pd
import numpy as np
import cv2
import chainer
from chainer import cuda
import chainer.serializers as S
from chainer import Variable

import util
from classifier import Classifier
from archs import ResNet101_c, slice_resnet101


def get_args(archs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int,
                        help='if -1, use cpu only')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='resnet_c',
                        help='Convnet architecture')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--img_list', required=True)
    parser.add_argument('--root', default='resized_tests', help='root directory of img files')
    parser.add_argument('--mean', default='mean.npy',
                        help='Path to the mean file (computed by compute_mean.py)')
    args = parser.parse_args()
    return args


def img_to_input(img, model, gpu):
    x_batch = np.ndarray(
        (1, 3, model.insize, model.insize), dtype=np.float32)
    x_batch[0] = img

    if gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
    x = chainer.Variable(x_batch, volatile=True)
    return x


def print_result(score, categories, top_k=10):
    prediction = zip(score.data[0].tolist(), categories)
    prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
    for rank, (score, name) in enumerate(prediction[:top_k], start=1):
        print('#%d | %s | %4.1f%%' % (rank, name, score * 100))


def main():
    archs = {
        'resnet_c': ResNet101_c.ResNet,
        'slice_resnet': slice_resnet101.SliceResNet
    }
    args = get_args(archs)
    model = archs[args.arch]()
    mean_image = np.load(args.mean)
    classifier = Classifier(gpu=args.gpu, model=model, initmodel=args.initmodel)
    normalize = False if classifier.use_caffemodel else True
    file_lists = pd.read_csv(args.img_list, delimiter='\t')
    # categories = pd.read_csv("data_cookpad/clf_category_master.tsv")
    # categories = np.loadtxt(args.label_file, str, delimiter="\t")
    df = pd.DataFrame()
    print('cropwidth', 256 - model.insize)
    for img_file in file_lists.file_name:
        img_file = osp.join(args.root, img_file)
        print('classify', img_file)
        img = util.load_image(path=img_file, crop_size=model.insize, normalize=normalize, mean_image=mean_image)
        x = img_to_input(img, model=model, gpu=args.gpu)
        score = classifier.predict(x)
        index = score.data[0].argmax(0)
        print img_file
        print index
        df = df.append([index], ignore_index=True)
        # print_result(score=score, categories=categories, top_k=10)

    df.to_csv('submit.csv')
    print 'Saved to submit.csv!!'


if __name__ == '__main__':
    main()
