import os, sys
import argparse
import pandas as pd
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', required=True,
                    help='path to input directory')
    parser.add_argument('-l', '--labels', required=True,
                    help='files of label list')
    parser.add_argument('-p', '--pairs', required=True,
                    help='file of pairs of label and file name')
    parser.add_argument('-o', '--out_dir', required=True,
                    help='path to output directory')
    args = parser.parse_args()

    return args


def assort_files_to_label_dir(in_dir, labels, pairs, out_dir):
    for category in labels['category_name']:
        tar_dir = os.path.join(out_dir, category)
        if not os.path.exists(tar_dir):
            os.mkdir(tar_dir)
    for index, pair in pairs.iterrows():
        src_file = os.path.join(in_dir, pair.file_name)
        dist_file = os.path.join(out_dir, labels.loc[pair.category_id].category_name, pair.file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dist_file)
            print('%s copying files %s to %s' % (index, src_file, dist_file))
        else:
            print(src_file + 'not found')


def main():
    args = get_args()
    labels = pd.read_csv(args.labels, delimiter='\t', index_col=1)
    pairs = pd.read_csv(args.pairs, delimiter='\t')
    assort_files_to_label_dir(args.in_dir, labels, pairs, args.out_dir)


if __name__=='__main__':
    main()
