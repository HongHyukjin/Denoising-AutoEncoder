import copy
import cv2 as cv

noisy = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\test\output\noisy_img\np.jpg', cv.IMREAD_COLOR)
denoised = cv.imread(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\test\output\denoised_img\dp.jpg', cv.IMREAD_COLOR)

img = copy.deepcopy(noisy)
img[110:140, 110:140, :] = denoised[110:140, 110:140, :]
cv.imwrite(r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\test\output\synthesize\sp.jpg', img)

