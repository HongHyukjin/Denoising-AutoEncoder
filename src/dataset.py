import numpy as np
import random
import tensorflow as tf
import cv2 as cv
import glob
import os

def next_batch(path, batch_size):
    length = len(glob.glob(os.path.join(path + "noisy_image/*")))
    idx = np.arange(0, length)
    #batch사이즈만큼 인덱스 선택
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    noisy_path = path + "noisy_image/image"
    reference_path = path + "reference/image_"
    noisy_image = []
    reference = []
    blind_num = []
    #batch사이즈 만큼 추가
    for num, i in enumerate(idx):
        blind = random.randint(0, 4)
        blind_num.append(blind)
        #print(noisy_path + str(i//5) + "(" + str(blind) + ").jpg")
        #print(reference_path + str(i//5) + "(" + str(blind) + ").jpg")
        a = cv.imread(noisy_path + str(i//5) + "(" + str(blind) + ").jpg")
        b = cv.imread(reference_path + str(i//5) + "(" + str(blind) + ").JPG")
        noisy_image.append(a)
        reference.append(b)

    noisy_image = np.asarray(noisy_image)
    reference = np.asarray(reference)
    noisy_image = noisy_image.astype("float32")
    reference = reference.astype("float32")
    return noisy_image, reference, blind_num



# noisy_image = [cv.imread(noisy_path + str(i) + ".jpg") for num, i in enumerate(idx)]
#     reference = [cv.imread(reference_path + str(i) + ".jpg") for num, i in enumerate(idx)]
#     noisy_image = np.asarray(noisy_image)
#     reference = np.asarray(reference)
#     noisy_image = noisy_image.astype("float32")
#     reference = reference.astype("float32")
#
#     return noisy_image, reference