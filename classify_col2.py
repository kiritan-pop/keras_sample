# coding: utf-8

import shutil
from keras.models import Sequential,Model,load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D,\
                        Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Reshape, Concatenate
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.utils import Sequence, plot_model
import os,glob,sys,json,random,cv2, threading
import numpy as np
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.utils import Sequence, plot_model
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import EarlyStopping, LambdaCallback
import multiprocessing
import os,glob,sys,json,random,cv2, threading
import numpy as np
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter

import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,
                                                  visible_device_list='0'
                                                  ))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


ImageFile.LOAD_TRUNCATED_IMAGES = True

# 同時実行プロセス数
process_count = multiprocessing.cpu_count()
graph = tf.get_default_graph()

STANDARD_SIZE = (512, 512)
model_path = 'clall_col.model'

def c_model():
    alpha=0.1
    dout=0.5
    model= Sequential()
    model.add(GaussianNoise(stddev=0.05,input_shape=(512, 512, 3)))
    model.add(Conv2D(filters=32,  kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dout))
    model.add(Conv2D(filters=64,  kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(filters=64,  kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dout))
    model.add(Conv2D(filters=128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(filters=128,  kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dout))
    model.add(Conv2D(filters=256,  kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(filters=256,  kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dout))
    model.add(Conv2D(filters=512,  kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(filters=512,  kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dout))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dout))
    model.add(Dense(9))
    model.add(Activation('softmax'))

    return model

if __name__ == "__main__":
    color = {}
    color['red'] = [1,0,0,0,0,0,0,0,0]
    color['blue'] = [0,1,0,0,0,0,0,0,0]
    color['green'] = [0,0,1,0,0,0,0,0,0]
    color['purple'] = [0,0,0,1,0,0,0,0,0]
    color['brown'] = [0,0,0,0,1,0,0,0,0]
    color['pink'] = [0,0,0,0,0,1,0,0,0]
    color['blonde'] = [0,0,0,0,0,0,1,0,0]
    color['white'] = [0,0,0,0,0,0,0,1,0]
    color['black'] = [0,0,0,0,0,0,0,0,1]
    print(color.keys())
    labels = list(color.keys())

    # モデルを読み込む
    with session.as_default():
        with graph.as_default():
            model = c_model()
            if os.path.exists(model_path):
                model.load_weights(model_path, by_name=False)

            i_dirs = []
            for p in os.listdir('colors/'):
                for f in os.listdir('colors/' + p + '/'):
                    i_dirs.append('colors/' + p + '/' + f)

            for image_path in i_dirs:
                img = np.asarray(Image.open(image_path).convert('RGB').resize(STANDARD_SIZE,Image.LANCZOS))
                img = (img-127.5)/127.5
                result = model.predict(np.array([img]))[0]
                # print('result=',result)

                rslt_dict = {}
                if max(result) > 0.7:
                    # print(max(result))
                    # print(np.where(result == max(result) )[0][0])
                    name = labels[np.where(result == max(result) )[0][0]]
                    image_path_af = 'temp2/' + name + '/' + image_path.rsplit('/')[-1]
                    shutil.copyfile(image_path, image_path_af)
                    print( labels[np.where(result == max(result) )[0][0]])
                else:
                    pass
                    # print( 'other')
