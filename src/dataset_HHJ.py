import numpy as np
import tensorflow as tf


def decode_example(seralized_example, shape):
    features = tf.parse_single_example(seralized_example,
                                       features={
                                           'image/noisy_img': tf.FixedLenFeature([],
                                                                                 dtype=tf.string, default_value=''),
                                           'image/reference': tf.FixedLenFeature([],
                                                                                 dtype=tf.string, default_value=''),
                                       }
                                       )

    h, w, in_c, out_c = shape

    noisy_img = tf.decode_raw(features['image/noisy_img'], tf.float32)
    reference = tf.decode_raw(features['image/reference'], tf.float32)
    noisy_img = tf.reshape(noisy_img, [h, w, in_c])
    reference = tf.reshape(reference, [h, w, out_c])

    noisy_img, reference = preprocess(noisy_img, reference)

    return noisy_img, reference


def next_batch_tensor(tfrecord_path, shape, batch_size=1,
                      shuffle_buffer=0, prefetch_size=1, repeat=0):
    ''' 다음 데이터를 출력하기 위한 텐서를 출력한다.
    Args:
      tfrecord_path  : 읽을 tfrecord 경로(---/---/파일이름.tfrecord)
      shape          : 높이, 폭, 입력 채널, 출력 채널의 시퀀스
                       ex) [65, 65, 66, 3] <-- h, w, in_c, out_c
      batch_size     : 미니배치 크기
      shuffle_buffer : 데이터 섞기 위한 버퍼 크기
      prefetch_size  : 모름. 그냥 1 씀
      repeat         : 데이터가 다 읽힌 경우 Exception이 발생한다.
                       이를 없애기 위해서는 몇 번 더 반복할지 정해줘야 한다.
    Returns:
      noisy_img      : noise가 있는 이미지 tensor
      reference      : noise가 없는 이미지 tensor(조금은 있겠지만..)
    '''

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda x: decode_example(x, shape))
    dataset = dataset.batch(batch_size)

    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    if prefetch_size > 0:
        dataset = dataset.prefetch(buffer_size=prefetch_size)
    if repeat > 0:
        dataset = dataset.repeat(repeat)

    iterator = dataset.make_one_shot_iterator()

    next_noise_image, next_reference = iterator.get_next()

    return next_noise_image, next_reference


def calc_grad(data):
    h, w, c = data.get_shape()

    dX = data[:, 1:, :] - data[:, :-1, :]
    dY = data[1:, :, :] - data[:-1, :, :]
    dX = tf.concat((tf.zeros([h, 1, c]), dX), axis=1)
    dY = tf.concat((tf.zeros([1, w, c]), dY), axis=0)

    return tf.concat((dX, dY), axis=-1)


def preprocess(noisy_img, reference):

    color = noisy_img[:, :, :3]
    normal = noisy_img[:, :, 12:15]
    albedo = noisy_img[:, :, 16:19]

    noisy_img = tf.concat(
        [color, normal, albedo], axis=-1)

    reference = reference[:, :, 0:3]


    return noisy_img, reference