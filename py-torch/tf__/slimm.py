import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

filename = '/result'
path = r'/home/kai/下载/DSC_4477.jpg'

image_raw_data = tf.gfile.FastGFile(path, 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype = tf.float32)

    for i in range(0,4):
        resized = tf.image.resize_images(img_data, [300,300], method=i)
        ss = resized.eval()
        plt.imshow(ss)
        plt.show()