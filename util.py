import os
from random import randint

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import zoom
from skimage.transform import resize
from chainer import link

def check_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def preprocess_image(image, crop_size, mean_image, normalize=True, random=True):
    # It applies following preprocesses:
    #     - Cropping (random or center rectangular)
    #     - Random flip
    #     - Scaling to [0, 1] value
    _, h, w = image.shape

    # if random:
    #     # Randomly crop a region and flip the image
    #     top = randint(0, h - crop_size - 1)
    #     left = randint(0, w - crop_size - 1)
    #     if randint(0, 1):
    #         image = image[:, :, ::-1]
    # else:
        # Crop the center
    #     # top = (h - crop_size) // 2
    #     # left = (w - crop_size) // 2
    # bottom = top + crop_size
    # right = left + crop_size

    image = image
    # image = image[:, top:bottom, left:right]
    # image -= mean_image[:, top:bottom, left:right]
    # if normalize:
    #     image /= 255
    return image

def load_image(path, crop_size, mean_image, normalize=True, random=True):
    f = Image.open(path)
    image = np.asarray(f, dtype=np.float32)
    if image.ndim == 2:
        # image is greyscale
        image = image[:, :, np.newaxis]
    image = image.transpose(2, 0, 1)
    image = preprocess_image(image=image, crop_size=crop_size, 
                mean_image=mean_image, normalize=normalize, random=random)
    return image

def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print 'Copy %s' % child.name
