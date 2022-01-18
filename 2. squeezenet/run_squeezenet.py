import pandas as pd
import numpy as np 
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
import keras.utils.np_utils as kutils
import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_first')
# Setup some keras variables
batch_size = 1024 # 128
nb_epoch = 1 # 100
img_rows, img_cols = 28, 28


mnist = tf.keras.datasets.mnist

# Normalize dataset
(x_train, y_train) , (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Setup of SqueezeNet (http://arxiv.org/abs/1602.07360), which offers similar performance
# to AlexNet, while using drastically fewer parameters. Tested on CIFAR10, it also performs
# well on MNIST problem

# Uses the latest keras 1.0.2 Functional API

input_layer = Input(shape=(28, 28, 1))

#conv 1
conv1 = Convolution2D(96, 3, 3, activation='relu', padding = 'same')(input_layer)

#maxpool 1
maxpool1 = MaxPooling2D(pool_size=(2,2), padding = 'same')(conv1)

#fire 1
fire2_squeeze = Convolution2D(16, 1, 1, activation='relu', padding = 'same')(maxpool1)
fire2_expand1 = Convolution2D(64, 1, 1, activation='relu', padding = 'same')(fire2_squeeze)
fire2_expand2 = Convolution2D(64, 3, 3, activation='relu', padding = 'same')(fire2_squeeze)
merge1 = tf.keras.layers.Concatenate(axis=1)([fire2_squeeze, fire2_squeeze])
fire2 = Activation("linear")(merge1)

#fire 2
fire3_squeeze = Convolution2D(16, 1, 1, activation='relu', padding = 'same')(fire2)
fire3_expand1 = Convolution2D(64, 1, 1, activation='relu', padding = 'same')(fire3_squeeze)
fire3_expand2 = Convolution2D(64, 3, 3, activation='relu', padding = 'same')(fire3_squeeze)
merge2 = tf.keras.layers.Concatenate(axis=1)([fire3_squeeze, fire3_squeeze])
fire3 = Activation("linear")(merge2)

#fire 3
fire4_squeeze = Convolution2D(32, 1, 1, activation='relu', padding = 'same')(fire3)
fire4_expand1 = Convolution2D(128, 1, 1, activation='relu', padding = 'same')(fire4_squeeze)
fire4_expand2 = Convolution2D(128, 3, 3, activation='relu', padding = 'same')(fire4_squeeze)
merge3 = tf.keras.layers.Concatenate(axis=1)([fire4_squeeze, fire4_squeeze])
fire4 = Activation("linear")(merge3)

#maxpool 4
maxpool4 = MaxPooling2D((2,2), padding = 'same')(fire4)

#fire 5
fire5_squeeze = Convolution2D(32, 1, 1, activation='relu', padding = 'same')(maxpool4)
fire5_expand1 = Convolution2D(128, 1, 1, activation='relu', padding = 'same')(fire5_squeeze)
fire5_expand2 = Convolution2D(128, 3, 3, activation='relu', padding = 'same')(fire5_squeeze)
merge5 = tf.keras.layers.Concatenate(axis=1)([fire5_squeeze, fire5_squeeze])
fire5 = Activation("linear")(merge5)

#fire 6
fire6_squeeze = Convolution2D(48, 1, 1, activation='relu', padding = 'same')(fire5)
fire6_expand1 = Convolution2D(192, 1, 1, activation='relu', padding = 'same')(fire6_squeeze)
fire6_expand2 = Convolution2D(192, 3, 3, activation='relu', padding = 'same')(fire6_squeeze)
merge6 = tf.keras.layers.Concatenate(axis=1)([fire6_squeeze, fire6_squeeze])
fire6 = Activation("linear")(merge6)

#fire 7
fire7_squeeze = Convolution2D(48, 1, 1, activation='relu', padding = 'same')(fire6)
fire7_expand1 = Convolution2D(192, 1, 1, activation='relu', padding = 'same')(fire7_squeeze)
fire7_expand2 = Convolution2D(192, 3, 3, activation='relu', padding = 'same')(fire7_squeeze)
merge7 = tf.keras.layers.Concatenate(axis=1)([fire7_squeeze, fire7_squeeze])
# merge7 = merge(inputs=[fire7_expand1, fire7_expand2], mode="concat", concat_axis=1)
fire7 =Activation("linear")(merge7)

#fire 8
fire8_squeeze = Convolution2D(64, 1, 1, activation='relu', padding = 'same')(fire7)
fire8_expand1 = Convolution2D(256, 1, 1, activation='relu', padding = 'same')(fire8_squeeze)
fire8_expand2 = Convolution2D(256, 3, 3, activation='relu', padding = 'same')(fire8_squeeze)
merge8 = tf.keras.layers.Concatenate(axis=1)([fire8_squeeze, fire8_squeeze])
#merge8 = merge(inputs=[fire8_expand1, fire8_expand2], mode="concat", concat_axis=1)
fire8 = Activation("linear")(merge8)

#maxpool 8
maxpool8 = MaxPooling2D((2,2), padding = 'same')(fire8)

#fire 9
fire9_squeeze = Convolution2D(64, 1, 1, activation='relu', padding = 'same')(maxpool8)
fire9_expand1 = Convolution2D(256, 1, 1, activation='relu', padding = 'same')(fire9_squeeze)
fire9_expand2 = Convolution2D(256, 3, 3, activation='relu', padding = 'same')(fire9_squeeze)
merge9 = tf.keras.layers.Concatenate(axis=1)([fire9_squeeze, fire9_squeeze])
#merge8 = merge(inputs=[fire9_expand1, fire9_expand2], mode="concat", concat_axis=1)
fire9 = Activation("linear")(merge8)
fire9_dropout = Dropout(0.5)(fire9)

#conv 10
conv10 = Convolution2D(10, 1, 1, padding = 'same')(fire9_dropout)

# The original SqueezeNet has this avgpool1 as well. But since MNIST images are smaller (1,28,28)
# than the CIFAR10 images (3,224,224), AveragePooling2D reduces the image size to (10,0,0), 
# crashing the script.

#avgpool 1
#avgpool10 = AveragePooling2D((13,13))(conv10)

flatten = Flatten()(conv10)

softmax = Dense(10, activation="softmax")(flatten)

model = Model(inputs=input_layer, outputs=softmax)

# Describe the model, and plot the model
model.summary()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Adjust model parameters to minimize the loss and train it
model.fit(x_train, y_train, epochs=20)

# Evaluate model performance
model.evaluate(x_test, y_test, verbose=2)

tf.keras.models.save_model(model, './')