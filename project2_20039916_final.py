

from scipy import misc
import imageio

import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


import cv2
import matplotlib.pyplot as plt

import numpy as np
from numpy import genfromtxt

work_path = "D:/_Lun/GoogleDrive/Information/HKUST/MSBD/MSBD 6000B Deep Learning/HW 2/data/"




X_path_train = np.genfromtxt(work_path + "train.txt", delimiter=' ' , usecols=[0] ,dtype=None )
y_path_train = np.genfromtxt(work_path + "train.txt", delimiter=' ' , usecols=[1] ,dtype=None )

X_path_val = np.genfromtxt(work_path + "val.txt", delimiter=' ' , usecols=[0] ,dtype=None )
y_path_val = np.genfromtxt(work_path + "val.txt", delimiter=' ' , usecols=[1] ,dtype=None )

X_path_test = np.genfromtxt(work_path + "test.txt", delimiter=' ' , usecols=[0] ,dtype=None )


def batch_utf8_path_add_work_path (work_path , sub_path):
    X_to_list = np.ndarray.tolist(sub_path)
    X_sub_path = []
    X_work_path = []
    for i in range(len(X_to_list)):
        X_sub_path.append(str(X_to_list[i], "utf-8")[2:])
        X_work_path.append(work_path)
    path  = [x + y for x, y in zip(X_work_path, X_sub_path)]
    return(path)

X_path_train = batch_utf8_path_add_work_path (work_path , X_path_train)
X_path_val = batch_utf8_path_add_work_path (work_path , X_path_val)
X_path_test = batch_utf8_path_add_work_path (work_path , X_path_test)

X_path = X_path_train + X_path_val
y = np.concatenate((y_path_train,y_path_val))


y_train = y



IMAGE_HEIGHT  = 50
IMAGE_WIDTH   = 50
NUM_CHANNELS  = 3



X_train = np.array([cv2.resize(cv2.imread (X_path[i]),(IMAGE_HEIGHT,IMAGE_WIDTH), interpolation = cv2.INTER_AREA) for i in range(0,len(X_path))], dtype=np.float64)
X_test = np.array([cv2.resize(cv2.imread (X_path_test[i]),(IMAGE_HEIGHT,IMAGE_WIDTH), interpolation = cv2.INTER_AREA) for i in range(0,len(X_path_test))], dtype=np.float64)


import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
K.set_image_dim_ordering('tf')
seed = 7
np.random.seed(seed)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# Create the model9
def cnn():
    model = Sequential()
    model.add(Convolution2D(128, 3, 3, input_shape=(50, 50, 3), activation= 'relu' , border_mode= 'same' ))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation= 'relu' , border_mode= 'same' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation= 'relu' , W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation= 'relu' , W_constraint=maxnorm(2)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation= 'relu' , W_constraint=maxnorm(1)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation= 'softmax' ))
    model.compile(loss= 'categorical_crossentropy' , optimizer='adam', metrics=[ 'accuracy' ])
    return model;

epochs = 200
estimator = KerasClassifier(build_fn=cnn)


estimator.fit(X_train, y_train, nb_epoch=epochs, batch_size = 64)
y_pred = estimator.predict(X_test)

dummy_y = np_utils.to_categorical(y_pred)

np.savetxt("project2_20039916.csv", y_pred , fmt = "%i" , delimiter=",")
