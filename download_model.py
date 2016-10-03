#!/usr/bin/env python
from __future__ import print_function
import argparse
import zipfile

import six

def get_args()
    parser = argparse.ArgumentParser(
        description='Download a Caffe reference model')
    parser.add_argument('model_type',
                        choices=('alexnet', 'caffenet', 'googlenet', 'resnet'),
                        help='Model type (alexnet, caffenet, googlenet, resnet)')
    args = parser.parse_args()
    return args

def get_url_filename(model_type):
    if model_type == 'alexnet':
        url = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
        name = 'bvlc_alexnet.caffemodel'
    elif model_type == 'caffenet':
        url = 'http://dl.caffe.berkeleyvision.org/' \
              'bvlc_reference_caffenet.caffemodel'
        name = 'bvlc_reference_caffenet.caffemodel'
    elif model_type == 'googlenet':
        url = 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
        name = 'bvlc_googlenet.caffemodel'
    elif model_type == 'resnet':
        url = 'http://research.microsoft.com/en-us/um/people/kahe/resnet/' \
              'models.zip'
        name = 'models.zip'
    else:
        raise RuntimeError('Invalid model type. Choose from '
                           'alexnet, caffenet, googlenet and resnet.')
    return url, name

def download_modelfile(url, name, model_type):
    print('Downloading model file...')
    six.moves.urllib.request.urlretrieve(url, name)
    if model_type == 'resnet':
        with zipfile.ZipFile(name, 'r') as zf:
            zf.extractall('.')
    print('Done')

def main():
    args = get_args()
    print(args.model_type)
    url, name = get_url_filename(model_type=args.model_type)
    download_modelfile(url, name, model_type=args.model_type)

if __name__ == '__main__':
    main()
