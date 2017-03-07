import argparse
import os, sys
import logging

import cv2
import numpy as np
import pandas as pd

import util 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_files', nargs='+', required=True,
                    help='Path to img to resize or image-label file path')
    parser.add_argument('-o', '--out_dirpath', default='./resized/',
                    help='Path to the output img dir path')
    parser.add_argument('-r', '--rename', default=0, type=int,
                    help='if 1, rename output img file name')
    parser.add_argument('-l', '--label', default=0, type=int,
                    help='if 1, make img-label list file in current dir')
    parser.add_argument('--out_imglabel',  
                    help="img-label pair txtfile's path")
    args = parser.parse_args()
    return args

def extract_imgpath(imglabel_path):
    df = pd.read_csv(imglabel_path, sep='\s+', names=['img_path','label'])
    return list(df.img_path), list(df.label)

def get_img_label(img_files):
    if 'txt' in img_files[0]:
        img_files, label = extract_imgpath(imglabel_path=img_files[0])
    else:
        label = range(len(img_files)) # dummy
    return img_files, label

def resize(img, target_shape=(256,256)):
    height, width, depth = img.shape
    x = max(height, width)
    new_img = np.zeros((x, x, depth)).astype(np.uint8)
    offset = ((np.array(new_img.shape) - np.array(img.shape)) / 2).astype(np.int)
    new_img[offset[0]:offset[0]+img.shape[0], offset[1]:offset[1]+img.shape[1], :] = img
    cropped_img = cv2.resize(new_img, target_shape)
    return cropped_img

def write_imglabel_list(imglabel_list, out_filepath):
    """
    write img-label list file as typ.txt
    """
    with open(out_filepath, "w") as fw:
        for img_path, label in imglabel_list:
            fw.write('{0} {1}\n'.format(img_path, label))
        logging.info('wrote {0}'.format(out_filepath))

def resize_imgs(img_files, label, out_dirpath, rename=False, make_imglabel=False, out_imglabel='trainlabel_pair.txt'):
    util.check_dirs(out_dirpath)
    if make_imglabel:
        imglabel_list = []
    for rename_num, (img_path, label) in enumerate(zip(img_files, label)):
        #read and crop image
        print('img_path',img_path)
        if img_path.find('.jpg') == -1:
            continue
        img = cv2.imread(img_path)
#        cropped_img = resize(img, target_shape=(256,256))
        cropped_img = img

        #save image
        if rename:
            save_path = os.path.join(out_dirpath, '{0}.jpg'.format(rename_num))
        else:
            save_path = os.path.join(out_dirpath, os.path.basename(img_path))
        print('save path is {0}'.format(save_path))
        cv2.imwrite(save_path, cropped_img)

        #make image-label txtfile
        if make_imglabel:
            imglabel_list.append((save_path, label))

    #save image-label txtfile
    if make_imglabel:
        write_imglabel_list(imglabel_list, out_filepath=out_imglabel)

def main():
    args = get_args()
    img_files, label = get_img_label(img_files=args.img_files)
    resize_imgs(img_files, label, out_dirpath=args.out_dirpath,
            rename=args.rename, make_imglabel=args.label, out_imglabel=args.out_imglabel)

if __name__ == '__main__':
    main()
