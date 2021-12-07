import os
import tensorflow as tf
import src.utils as utils
from itertools import count
from src.model import DenoisingNet
from src.dataset import next_batch
from src.base_model import SSIM
import cv2 as cv
import matplotlib.pyplot as plt

#GPU 할당 문제 해결 코드
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# 모델의 이름에 따라 checkpoint나 summary 폴더 저장함.(debug 폴더에서)
MODEL_NAME   = 'DAE6'

# checkpoint와 inference 결과를 저장할 폴더
DEBUG_DIR         = './debug/'

# 학습 중 이미지가 잘 나오는지 보기 위한 폴더
DEBUG_IMAGE_DIR   = DEBUG_DIR + 'images/' + MODEL_NAME + '/'
NOISY_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "noisy_img/")
REFER_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "reference/")
DENOISED_IMG_DIR  = os.path.join(DEBUG_IMAGE_DIR, "denoised_img/")

# 모델 저장하기 위한 폴더
CKPT_DIR          = DEBUG_DIR + 'checkpoint/' + MODEL_NAME + '/'

# 하이퍼 파라미터
BATCH_SIZE        = 8

INPUT_CH          = 3     # model에 입력으로 들어갈 채널 수
OUTPUT_CH         = 3     # 모델의 출력으로 나올 채널 수 (color 3)

TRAIN_PATH    = './data/train2/'
VALID_PATH    = './data/valid2/'

SAVE_PERIOD       = 1000   # 이 주기 마다 모델 가중치 저장
VALID_PERIOD      = 50   # 이 주기 마다 검증(검증 데이터로 inference하고 이미지 저장)


def main():

  # Denoising Model. src\base_model.py과 src\model.py를 참조할 것

  net = DenoisingNet(input_shape=[None, None, INPUT_CH],
                    output_shape=[None, None, OUTPUT_CH],
                    loss_func='SSIM',
                    start_lr=1e-7,
                    lr_decay_step=10000,
                    lr_decay_rate=1.0)

  sess = tf.Session(config=tf_config)

  # debug용 폴더들 생성
  utils.make_dir(CKPT_DIR)
  utils.make_dir(NOISY_IMAGE_DIR)
  utils.make_dir(REFER_IMAGE_DIR)
  utils.make_dir(DENOISED_IMG_DIR)

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


  # 학습 시작 Epoch
  for epoch in count():

    # 학습용 데이터 tensor 생성 src\dataset 참조
    train_noisy_img, train_reference = \
      next_batch(
        path = TRAIN_PATH,
        batch_size = BATCH_SIZE
      )

    # 학습용 데이터 다 읽을 때 까지
    while True:
      # 더이상 읽을 것이 없으면 빠져 나오고 다음 epoch으로
      try:
        # 학습용 데이터를 위에 설정한 베치사이즈만큼 가져온다.
        noisy_img, reference = \
            sess.run([train_noisy_img, train_reference])

      except tf.errors.OutOfRangeError as e:
        print("Done")
        break

      # 데이터로 학습
      _, step, lr = sess.run([net.train_op, net.global_step, net.lr],
                             feed_dict={net.inputs: noisy_img,
                                        net.refers: reference})

      if step % 100 == 0:
        print(step)
      #가중치 저장
      if step % SAVE_PERIOD == SAVE_PERIOD - 1:
        saver.save(sess, os.path.join(CKPT_DIR, "model.ckpt"), global_step = step + 1,write_meta_graph=False)

      #Inference
      if step % VALID_PERIOD == VALID_PERIOD - 1:
        # 검증용 데이터 tensor 생성 src\dataset 참조
        # =========================================================================
        valid_noisy_img, valid_reference = \
          next_batch(
            path=VALID_PATH,
            batch_size=1
          )
        # =========================================================================

        noisy_img, reference = \
            sess.run([valid_noisy_img, valid_reference])

        loss, denoised_img = sess.run([net.loss, net.outputs],
                                              feed_dict={net.inputs:noisy_img,
                                                         net.refers:reference})

        print(" Test ] Loss ", loss)


        # validation으로 testing 해보는 코드

        denoised_img = denoised_img.reshape(256,256,3)
        reference_img = reference.reshape(256,256,3)
        noisy_img = noisy_img.reshape(256,256,3)

        rp = REFER_IMAGE_DIR + f'{step + 1}'
        dp = DENOISED_IMG_DIR + f'{step + 1}'
        np = NOISY_IMAGE_DIR + f'{step + 1}'

        #test 이미지 저장
        cv.imwrite(rp + ".jpg", reference_img)
        cv.imwrite(dp + ".jpg", denoised_img)
        cv.imwrite(np + ".jpg", noisy_img)


  sess.close()
  tf.reset_default_graph()



if __name__ == '__main__':
  main()