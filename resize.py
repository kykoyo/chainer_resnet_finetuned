
import cv2
import argparse
import os, sys
import pandas as pd
import logging

 
def check_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def extract_imgpath(imglabel_path):
    df = pd.read_csv(imglabel_path, sep='\s+', names=['img_path','label'])
    return list(df.img_path), list(df.label)

def write_imglabel_list(imglabel_list, out_filepath):
    """
    write img-label list file as typ.txt
    """
    with open(out_filepath, "w") as fw:
        for img_path, label in imglabel_list:
            fw.write('{0} {1}\n'.format(img_path, label))
        logging.info('wrote {0}'.format(out_filepath))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_files', nargs='+', required=True,
                    help='Path to img to resize or image-label file path')
    parser.add_argument('-o', '--out_dirpath', default='./resized/',
                    help='Path to the output img dir path')
    parser.add_argument('-r', '--rename', default=0, type=int,
                    help='if 1, rename output img file name')
    parser.add_argument('-l', '--label', default=0, type=int,
                    help='if 1, make img-label list file in current dir')
    args = parser.parse_args()

    if 'txt' in args.img_files[0]:
        img_files, label = extract_imgpath(args.img_files[0])
        imglabel_list = []
    else:
        img_files = args.img_files
        label = range(len(img_files)) # dummy
        
    check_dirs(args.out_dirpath)
    print('len(img_files)', len(img_files))
    for rename_num, (img_path, label) in enumerate(zip(img_files, label)):
        print('img_path',img_path)
        target_shape = (256, 256)
        if img_path.find('.jpg') == -1:
            continue

        img = cv2.imread(img_path)
        height, width, depth = img.shape
        output_side_length=256
        new_height = output_side_length
        new_width = output_side_length
        if height > width:
            new_height = output_side_length * height / width
        else:
            new_width = output_side_length * width / height
        resized_img = cv2.resize(img, (new_width, new_height))
        height_offset = (new_height - output_side_length) / 2
        width_offset = (new_width - output_side_length) / 2
        cropped_img = resized_img[height_offset:height_offset + output_side_length,
        width_offset:width_offset + output_side_length]

        if args.rename==1:
            save_path = os.path.join(args.out_dirpath, '{0}.jpg'.format(rename_num))
        else:
            save_path = os.path.join(args.out_dirpath, os.path.basename(img_path))
        print('save path is {0}'.format(save_path))
        cv2.imwrite(save_path, cropped_img)

        if args.label==1:
            imglabel_list.append((save_path, label))

    if args.label==1:
        write_imglabel_list(imglabel_list, out_filepath='./trainlabel_pairs.txt')

if __name__ == '__main__':
    main()
