import cv2 as cv
import glob
import os
import shutil

path = r'E:\HyukJin\2021\JongSul\LeeDaeWon\Denoising AutoEncoder\data\valid7\reference'
file_names = os.listdir(path)
print(file_names)
#print(file_names[0][:-4])

#폴더 내 이미지 이름 변경
#for idx, file in enumerate(file_names):
    #print(file)
#    src = os.path.join(path, file)
#    dst = 'image_' + str(idx) + '.JPG'
#    dst = os.path.join(path, dst)
#    os.rename(src, dst)


#폴더 내 이미지 X 5
idx = 0
for name in file_names:
    src = os.path.join(path, name)
    dst = src[:-4] + '(0).JPG'
    img = cv.imread(src)
    os.rename(src, dst)
    for i in range(1, 5):
        cv.imwrite(src[:-4] + '(' + str(i) + ').JPG', img)
    idx += 1



