import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='make txt file of labels')
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    caltech = os.listdir(args.input_dir)
    sorted_caltech = sorted(caltech, key=str.lower)
    with open(args.output_file, "w") as fw:
        for label in sorted_caltech:
            fw.write('{0}\n'.format(label))

if __name__=='__main__':
    main()
