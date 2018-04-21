# coding: utf-8

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
                                                  visible_device_list='2'
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

class Generator(Sequence):
    def __init__(self, batch_size):
        # コンストラクタ
        self.color_path = 'class_col/'
        self.batch_size = batch_size
        self.color_images = []
        for p in os.listdir(self.color_path):
            for f in os.listdir(self.color_path + p + '/'):
                self.color_images.append(self.color_path + p + '/' + f)
        self.sample_per_epoch = int(len(self.color_images)/self.batch_size)  #端数は切り捨て。端数分は・・・

        self.color = {}
        self.color['red'] = [1,0,0,0,0,0,0,0,0]
        self.color['blue'] = [0,1,0,0,0,0,0,0,0]
        self.color['green'] = [0,0,1,0,0,0,0,0,0]
        self.color['purple'] = [0,0,0,1,0,0,0,0,0]
        self.color['brown'] = [0,0,0,0,1,0,0,0,0]
        self.color['pink'] = [0,0,0,0,0,1,0,0,0]
        self.color['blonde'] = [0,0,0,0,0,0,1,0,0]
        self.color['white'] = [0,0,0,0,0,0,0,1,0]
        self.color['black'] = [0,0,0,0,0,0,0,0,1]

    def __getitem__(self, idx):
        x = []
        y = []
        # データの取得実装
        ytmp1 = self.color_images[self.batch_size*idx:self.batch_size*(idx+1)]
        for i,t in enumerate(ytmp1):
            #本物
            img = Image.open(t)
            img = img.convert('RGB')
            org = img
            img = img.resize(STANDARD_SIZE,Image.LANCZOS)
            x.append( (np.asarray(img)-127.5)/127.5)
            y.append( np.asarray(self.color[ t.rsplit('/')[-2] ]) )
            if idx % 50 == 0:
                filename = 'temp_col/{0:08d}_{1:1d}_{2}.png'.format(idx,i,t.rsplit('/')[-2])
                org.convert('RGB').resize((np.asarray(org).shape[1],np.asarray(org).shape[0]),Image.LANCZOS).save(filename,'png', optimize=True)
        return np.asarray(x), np.asarray(y)

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！
        return self.sample_per_epoch

def d_on_epoch_end(epoch, logs):
    th = threading.Thread(target=model.save_weights, args=(model_path,) )
    th.start()

def d_main(idx, d_generator):
    print('＊＊＊＊d_main＊＊＊＊')

if __name__ == "__main__":
    # モデルを読み込む
    with session.as_default():
        with graph.as_default():
            model = c_model()
            if os.path.exists(model_path):
                model.load_weights(model_path, by_name=False)

            model.compile(loss='categorical_crossentropy',
                            optimizer=Adam(lr=1e-5, beta_1=0.5),
                            metrics=['accuracy']
                            )
            d_generator = Generator(batch_size=4)
            model.fit_generator(
                    d_generator,
                    callbacks=[LambdaCallback(on_epoch_end=d_on_epoch_end)],
                    steps_per_epoch=1024,
                    epochs=1024,
                    validation_data=d_generator,
                    validation_steps=24,
                    max_queue_size=process_count*5,
                    workers=1,
                    use_multiprocessing=False)
