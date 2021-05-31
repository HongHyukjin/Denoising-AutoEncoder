import numpy as np
import tensorflow as tf
import cv2 as cv
import glob
import os

def next_batch(path, batch_size):
    noisy_image = [cv.imread(file) for file in glob.glob(os.path.join(path + "/noisy_image"))]
    reference = [cv.imread(file) for file in glob.glob(os.path.join(path + "/reference"))]
    idx = np.arange(0, len(noisy_image))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    noisy_image = [noisy_image[i] for i in idx]
    reference = [reference[i] for i in idx]

    return np.asarray(noisy_image), np.asarray(reference)



def preprocess(noisy_img, reference):

    color = noisy_img[:, :, :3]
    normal = noisy_img[:, :, 12:15]
    albedo = noisy_img[:, :, 16:19]

    noisy_img = tf.concat(
        [color, normal, albedo], axis=-1)

    reference = reference[:, :, 0:3]


    return noisy_img, reference