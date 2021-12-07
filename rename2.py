import os
import numpy as np
import cv2 as cv
import glob
import random
import copy

path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\train7\reference'

file_list = os.listdir(path)


#해당 폴더의 파일을 순서대로 읽음
for idx, file in enumerate(file_list):
    #print(file)
    src = os.path.join(path, file)
    dst = 'image_' + str(idx) + '.JPG'
    dst = os.path.join(path, dst)
    os.rename(src, dst)







