#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import glob
import logging
import os
import random
import shutil

import util

def check_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def write_imglabel_list(imglabel_list, out_filepath):
    """
    write img-label list file as typ.txt
    """
    with open(out_filepath, "w") as fw:
        for img_path, label in imglabel_list:
            fw.write('{0} {1}\n'.format(img_path, label))
        logging.info('wrote {0}'.format(out_filepath))

def separate_train_val(args):
    check_dirs(args.output_dir)
    check_dirs(os.path.join(args.output_dir, 'train'))
    check_dirs(os.path.join(args.output_dir, 'val'))
    directories = os.listdir(args.root)
    categories_sorted = sorted(directories, key=str.lower)
    cate_label_dict = {}
    for label, cate in enumerate(categories_sorted):
        cate_label_dict[cate] = label
    train_imglabel_list = []
    val_imglabel_list = []

    for dir_index, dir_name in enumerate(categories_sorted):
        files = glob.glob(os.path.join(args.root, dir_name, '*.jpg'))
        random.shuffle(files)
        if len(files) == 0:
            continue

        for file_index, file_path in enumerate(files):
            if file_index % args.val_freq != 0:
                target_dir = os.path.join(args.output_dir, 'train', dir_name)
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)
                shutil.copy(file_path, target_dir)
                copied_file_path = os.path.join(target_dir, os.path.basename(file_path))
                train_imglabel_list.append((copied_file_path, cate_label_dict[dir_name]))
                logging.info('Copied {} => {}'.format(file_path, target_dir))
            else:
                target_dir = os.path.join(args.output_dir, 'val', dir_name)
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)
                shutil.copy(file_path, target_dir)
                copied_file_path = os.path.join(target_dir, os.path.basename(file_path))
                val_imglabel_list.append((copied_file_path, cate_label_dict[dir_name]))
                logging.info('Copied {} => {}'.format(file_path, target_dir))

    write_imglabel_list(imglabel_list=train_imglabel_list, out_filepath=args.output_train_imglabel_file)
    write_imglabel_list(imglabel_list=val_imglabel_list, out_filepath=args.output_val_imglabel_file)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='converter')
    parser.add_argument('--root', default='.')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--output_train_imglabel_file', default='./train_label.txt')
    parser.add_argument('--output_val_imglabel_file', default='./val_label.txt')
    parser.add_argument('--val_freq', type=int, default=10)
    args = parser.parse_args()

    separate_train_val(args)

if __name__ == '__main__':
    main()
 
