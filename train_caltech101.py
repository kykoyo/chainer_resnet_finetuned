#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import random
import cPickle as pickle
import magic
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

from archs import alex, googlenet, googlenet_c, googlenetbn, nin, vgg, ResNet101, ResNet101_c
import util

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        image, label = self.base[i]
        image = util.preprocess_image(image=image, crop_size=self.crop_size,
                    mean_image=self.mean, normalize=True, random=self.random)
        return image, label

def get_args(archs):
    parser = argparse.ArgumentParser(description='Learning convnet from ILSVRC2012 dataset')

    parser.add_argument('train', help='Path to training image-label list file') #訓練用のラベルリスト
    parser.add_argument('val', help='Path to validation image-label list file') #テスト用のラベルリスト
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture') #ネットワークの種類
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size') #ミニバッチのサイズ
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train') #エポック数
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU') #使用するGPU
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file') #フィアンチューニングを使用する際、出力のクラス数が変わる場合 .pkl
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes') #並行してロードするプロセス数
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)') #平均画像ファイル
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file') #ファインチューニングを使用する際、出力数が変わらない場合 .h5
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory') #出力ディレクトリ
    parser.add_argument('--output_model',  default='final_model.h5',
                        help='Output model name') #出力モデル
    parser.add_argument('--output_optimizer',  default='final_optimizer.h5',
                        help='Output optimizer name') #出力最適化名
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files') #画像ファイルのルートディレクトリ
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size') #検証時のミニバッチのサイズ
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    return args

def main():
    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenet_c': googlenet_c.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        'vgg': vgg.VGG,
        'resnet': ResNet101.ResNet,
        'resnet_c': ResNet101_c.ResNet
    }
    args = get_args(archs)

    # Initialize the model to train
    model = archs[args.arch]()
    initmodel = archs['resnet']()
    if args.initmodel:
        print('Initializing the model')
        file_type = magic.from_file(args.initmodel, mime=True)
        if 'hdf' in file_type:
            chainer.serializers.load_hdf5(args.initmodel, initmodel)
        elif 'zip' in file_type:
            chainer.serializers.load_npz(args.initmodel, initmodel)
        util.copy_chainermodel(initmodel, model)
        # print('Load model from', args.initmodel)
        # initmodel = pickle.load(open(args.initmodel))
        # util.copy_chainermodel(initmodel, model)
        # # chainer.serializers.load_hdf5(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = PreprocessedDataset(args.train, args.root, mean, model.insize)
    val = PreprocessedDataset(args.val, args.root, mean, model.insize, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.00005)  # パラメータの学習方法は慣性項付きの確率的勾配法で, 学習率は0.0005に設定.
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))  # l2正則化

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'

    # Copy the chain with shared parameters to flip 'train' flag only in test
    eval_model = model.copy()
    eval_model.train = False

    def lr_shift():  # DenseNet specific!
        if updater.epoch == 100 or updater.epoch == 125:
            optimizer.lr *= 0.1
        return optimizer.lr

    trainer.extend(extensions.Evaluator(val_iter, eval_model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.observe_value(
        'lr', lambda _: lr_shift()), trigger=(1, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        print('Load optimizer state from', args.resume)
        chainer.serializers.load_hdf5(args.resume, trainer)

    trainer.run()

    print('Saving model...')
    chainer.serializers.save_hdf5(os.path.join(args.out, args.output_model), model)
    print('Saving trainer...')
    chainer.serializers.save_hdf5(os.path.join(args.out, args.output_optimizer), trainer)

if __name__ == '__main__':
    main()
