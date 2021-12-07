#SSIM를 비교하는 TEST
from src.base_model import SSIM
import tensorflow as tf
import numpy as np
import cv2 as cv

reference = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\test\output\reference\rp.jpg')
noisy = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\test\output\noisy_img\np.jpg')
denoised = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\test\output\denoised_img\dp.jpg')
#synthesize = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\test\output\synthesize\sp.jpg')
#reference = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\debug\images\DAE12\denoised_img\rf_test.jpg')
#noisy = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\test\noisy_img\no.jpg')
#denoised = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\test\denoised_img\de_test.jpg')

#print(reference)
reference = reference.reshape(1, 256, 256, 3)
reference = reference.astype('float32')
#print(noisy)
noisy = noisy.reshape(1, 256, 256, 3)
noisy = noisy.astype('float32')
#print(denoised)
denoised = denoised.reshape(1, 256, 256, 3)
denoised = denoised.astype('float32')
#print(denoised)

#synthesize = synthesize.reshape(1, 256, 256, 3)
#synthesize = synthesize.astype('float32')

result1 = SSIM(noisy, reference)
result2 = SSIM(denoised, reference)
#result3 = SSIM(synthesize, reference)

sess = tf.Session()
result1 = sess.run(result1)
result2 = sess.run(result2)
#result3 = sess.run(result3)
print("Reference와 Inpainting image의 SSIM : ", end="")
print(1 - result1)
print("Reference와 Denoising image의 SSIM : ", end="")
print(1 - result2)
#print("Reference와 Synthesize image의 SSIM : ", end="")
#print(1 - result3)