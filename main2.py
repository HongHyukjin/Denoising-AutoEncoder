import os
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

#최종 : DAE15
# 모델의 이름에 따라 checkpoint 폴더 저장함.
MODEL_NAME   = 'DAE19'

# checkpoint와 inference 결과를 저장할 폴더
DEBUG_DIR         = './debug/'

# 학습 중 이미지가 잘 나오는지 보기 위한 폴더
DEBUG_IMAGE_DIR   = DEBUG_DIR + 'images/' + MODEL_NAME + '/'
NOISY_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "noisy_img/")
REFER_IMAGE_DIR   = os.path.join(DEBUG_IMAGE_DIR, "reference/")
DENOISED_IMG_DIR  = os.path.join(DEBUG_IMAGE_DIR, "denoised_img/")
SYNTHESIZE_IMG_DIR  = os.path.join(DEBUG_IMAGE_DIR, "synthesize/")

# 모델 저장하기 위한 폴더
CKPT_DIR          = DEBUG_DIR + 'checkpoint/' + MODEL_NAME + '/'

# 하이퍼 파라미터
BATCH_SIZE        = 4

INPUT_CH          = 3     # model에 입력으로 들어갈 채널 수
OUTPUT_CH         = 3     # 모델의 출력으로 나올 채널 수 (color 3)

TRAIN_PATH    = './data/train7/'
VALID_PATH    = './data/valid7/'

SAVE_PERIOD       = 100   # 이 주기 마다 모델 가중치 저장
VALID_PERIOD      = 50   # 이 주기 마다 검증(검증 데이터로 inference하고 이미지 저장)
SYNTHESIZE_PERIOD = 50


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


  # 학습 시작
  while(True):
    # 학습용 데이터 생성
    train_noisy_img, train_reference, blind_num = \
      next_batch(
        path = TRAIN_PATH,
        batch_size = BATCH_SIZE
      )
    #print(train_noisy_img.shape)
    #print(train_reference.shape)

    # 데이터로 학습
    _, step, lr = sess.run([net.train_op, net.global_step, net.lr],
                             feed_dict={net.inputs: train_noisy_img,
                                        net.refers: train_reference})

    if step % 50 == 0:
      print(step)

    #가중치 저장
    if step % SAVE_PERIOD == SAVE_PERIOD - 1:
      saver.save(sess, os.path.join(CKPT_DIR, "model.ckpt"), global_step = step + 1,write_meta_graph=False)

    #Inference
    if step % VALID_PERIOD == VALID_PERIOD - 1:
      # 검증용 데이터 생성
      # =========================================================================
      valid_noisy_img, valid_reference, blind_num = \
        next_batch(
          path=VALID_PATH,
          batch_size=1
        )
      #print(valid_noisy_img.shape)
      #print(valid_reference.shape)
      print(blind_num)
      # =========================================================================
      loss, denoised_img = sess.run([net.loss, net.outputs],
                                              feed_dict={net.inputs:valid_noisy_img,

                                                         net.refers:valid_reference})

      denoised_img = denoised_img.astype('int32')
      denoised_img = denoised_img.astype('float32')

      print("SSIM + L1 ", loss)

      loss1 = sess.run(SSIM(denoised_img, valid_reference))
      loss2 = sess.run(SSIM(valid_noisy_img, valid_reference))

      print(" denoised_img - reference Loss ", loss1)
      print(" noisy_img - reference Loss ", loss2
            )

      #일정 step마다 noisy와 denoised 합성
      if step % SYNTHESIZE_PERIOD == SYNTHESIZE_PERIOD - 1:
          syn_img = synthesize(valid_noisy_img, denoised_img, blind_num)
          loss3 = sess.run(SSIM(syn_img, valid_reference))
          print(" synthesize_img - reference Loss ", loss3)
          syn_img = syn_img.reshape(256, 256, 3)
          sp = SYNTHESIZE_IMG_DIR + f'{step + 1}'
          cv.imwrite(sp + ".jpg", syn_img)


      #Inference 결과 저장
      denoised_img = denoised_img.reshape(256,256,3)
      reference_img = valid_reference.reshape(256,256,3)
      noisy_img = valid_noisy_img.reshape(256,256,3)

      rp = REFER_IMAGE_DIR + f'{step + 1}'
      dp = DENOISED_IMG_DIR + f'{step + 1}'
      np = NOISY_IMAGE_DIR + f'{step + 1}'


      cv.imwrite(rp + ".jpg", reference_img)
      cv.imwrite(dp + ".jpg", denoised_img)
      cv.imwrite(np + ".jpg", noisy_img)

  sess.close()


if __name__ == '__main__':
  main()