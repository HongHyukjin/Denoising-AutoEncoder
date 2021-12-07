import glob
import cv2 as cv
import tensorflow as tf
import os
from src.base_model import SSIM

noisy_path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\debug\images\DAE19\noisy_img'
syn_path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\debug\images\DAE19\synthesize'
ref_path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\debug\images\DAE19\reference'
den_path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\debug\images\DAE19\denoised_img'

#image들 각각의 이름 ex)1234.jpg
img_name_list = os.listdir(syn_path)
print(len(img_name_list))

noisy_sum = 0
denoised_sum = 0
synthesize_sum = 0

for idx, path in enumerate(img_name_list):

    noisy = cv.imread(noisy_path + '\\' + img_name_list[idx])
    reference = cv.imread(ref_path + '\\' + img_name_list[idx])
    synthesize = cv.imread(syn_path + '\\' + img_name_list[idx])
    denoised = cv.imread(den_path + '\\' + img_name_list[idx])

    noisy = noisy.reshape(1, 256, 256, 3)
    noisy = noisy.astype('float32')

    reference = reference.reshape(1, 256, 256, 3)
    reference = reference.astype('float32')

    denoised = denoised.reshape(1, 256, 256, 3)
    denoised = denoised.astype('float32')

    synthesize = synthesize.reshape(1, 256, 256, 3)
    synthesize = synthesize.astype('float32')

    result1 = SSIM(noisy, reference)
    result2 = SSIM(denoised, reference)
    result3 = SSIM(synthesize, reference)

    sess = tf.Session()
    result1 = 1 - sess.run(result1)
    result2 = 1 - sess.run(result2)
    result3 = 1 - sess.run(result3)

    print(idx)
    print(result1)
    print(result2)
    print(result3)

    noisy_sum += result1
    denoised_sum += result2
    synthesize_sum += result3

    print("noisy average : ", noisy_sum / (idx + 1))
    print("denoised_average : ", denoised_sum / (idx + 1))
    print("synthesize_average : ", synthesize_sum / (idx + 1))

    # print("noisy average : ", round(1 - noisy_average, 5))
    # print("denoised_average : ", round(1 - denoised_average, 5))
    # print("synthesize_average : ", round(1 - synthesize_average, 5))