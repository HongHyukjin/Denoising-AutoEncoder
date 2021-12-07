import os
import numpy as np
import tensorflow as tf
import src.utils as utils
from itertools import count
from src.model2 import DenoisingNet
from src.dataset import next_batch
from src.base_model import SSIM
import cv2 as cv


#GPU 할당 문제 해결 코드
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# 모델의 이름에 따라 checkpoint나 summary 폴더 저장함.(debug 폴더에서)
MODEL_NAME   = 'DAE11'

# checkpoint와 inference 결과를 저장할 폴더
DEBUG_DIR         = './debug/'
TEST_DIR          = './test/'
# 학습 중 이미지가 잘 나오는지 보기 위한 폴더
NOISY_IMAGE_DIR   = os.path.join(TEST_DIR, "noisy_img/")
REFER_IMAGE_DIR   = os.path.join(TEST_DIR, "reference/")
DENOISED_IMG_DIR  = os.path.join(TEST_DIR, "denoised_img/")

# 모델 저장하기 위한 폴더
CKPT_DIR          = DEBUG_DIR + 'checkpoint/' + MODEL_NAME + '/'

# 하이퍼 파라미터
BATCH_SIZE        = 8

INPUT_CH          = 3     # model에 입력으로 들어갈 채널 수
OUTPUT_CH         = 3     # 모델의 출력으로 나올 채널 수 (color 3)import os
import tensorflow as tf
import src.utils as utils
from src.model2 import DenoisingNet
from src.dataset import next_batch
from src.base_model import SSIM
import cv2 as cv
from synthesize import synthesize


#GPU 할당 문제 해결 코드
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

#최종 : DAE19
# 모델의 이름에 따라 checkpoint 폴더 저장함.
MODEL_NAME   = 'DAE18'

# checkpoint와 inference 결과를 저장할 폴더
DEBUG_DIR         = './debug/'
TEST_DIR          = './data/test'

# 학습 중 이미지가 잘 나오는지 보기 위한 폴더
NOISY_IMAGE_DIR   = os.path.join(TEST_DIR, "output/noisy_img/")
REFER_IMAGE_DIR   = os.path.join(TEST_DIR, "output/reference/")
DENOISED_IMG_DIR  = os.path.join(TEST_DIR, "output/denoised_img/")
SYNTHESIZE_IMG_DIR  = os.path.join(TEST_DIR, "output/synthesize/")

# 모델 저장하기 위한 폴더
CKPT_DIR          = DEBUG_DIR + 'checkpoint/' + MODEL_NAME + '/'

# 하이퍼 파라미터
BATCH_SIZE        = 1

INPUT_CH          = 3     # model에 입력으로 들어갈 채널 수
OUTPUT_CH         = 3     # 모델의 출력으로 나올 채널 수 (color 3)

TRAIN_PATH    = './data/test/'
VALID_PATH    = './data/test/'

SAVE_PERIOD       = 100   # 이 주기 마다 모델 가중치 저장
VALID_PERIOD      = 50   # 이 주기 마다 검증(검증 데이터로 inference하고 이미지 저장)
SYNTHESIZE_PERIOD = 100


def main():
  net = DenoisingNet(input_shape=[None, None, INPUT_CH],
                    output_shape=[None, None, OUTPUT_CH],
                    loss_func='SSIM+L1',
                    start_lr=2e-5,
                    lr_decay_step=10000,
                    lr_decay_rate=1.0)

  sess = tf.Session(config=tf_config)

  # debug용 폴더들 생성
  utils.make_dir(CKPT_DIR)
  utils.make_dir(NOISY_IMAGE_DIR)
  utils.make_dir(REFER_IMAGE_DIR)
  utils.make_dir(DENOISED_IMG_DIR)
  utils.make_dir(SYNTHESIZE_IMG_DIR)
  # Saver
  # =========================================================================
  saver = tf.train.Saver()

  print('Saver initialized')
  recent_ckpt_path = tf.train.latest_checkpoint(CKPT_DIR)

  if recent_ckpt_path is None:
    sess.run(tf.global_variables_initializer())
    print("Initializing variables...")
  else:
    saver.restore(sess, recent_ckpt_path)
    print("Restoring...", recent_ckpt_path)
  # =========================================================================

  valid_noisy_img, valid_reference, blind_num = \
    next_batch(
      path=VALID_PATH,
      batch_size=1
    )

  loss, denoised_img = sess.run([net.loss, net.outputs],
                                feed_dict={net.inputs: valid_noisy_img,

                                           net.refers: valid_reference})

  denoised_img = denoised_img.astype('int32')
  denoised_img = denoised_img.astype('float32')

  print("SSIM + L1 ", loss)

  loss1 = sess.run(SSIM(denoised_img, valid_reference))
  loss2 = sess.run(SSIM(valid_noisy_img, valid_reference))

  print(" denoised_img - reference Loss ", loss1)
  print(" noisy_img - reference Loss ", loss2
        )

  # Inference 결과 저장
  denoised_img = denoised_img.reshape(256, 256, 3)
  reference_img = valid_reference.reshape(256, 256, 3)
  noisy_img = valid_noisy_img.reshape(256, 256, 3)

  rp = REFER_IMAGE_DIR + 'rp'
  dp = DENOISED_IMG_DIR + 'dp'
  np = NOISY_IMAGE_DIR + 'np'

  cv.imwrite(rp + ".jpg", reference_img)
  cv.imwrite(dp + ".jpg", denoised_img)
  cv.imwrite(np + ".jpg", noisy_img)

  sess.close()


if __name__ == '__main__':
  main()

