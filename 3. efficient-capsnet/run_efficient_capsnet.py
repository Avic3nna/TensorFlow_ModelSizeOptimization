#https://colab.research.google.com/drive/1hEnjNiTRVkQczIvfIxskfoOTligNRjQG?usp=sharing#scrollTo=ikY_gNwqM9fA

#!git clone https://github.com/EscVM/Efficient-CapsNet.git
# move the working directory to Efficient-CapsNet folder
import os
os.chdir('Efficient-CapsNet')
#!pip install -r requirements.txt

import tensorflow as tf
from utils import AffineVisualizer, Dataset
from models import EfficientCapsNet

mnist_dataset = Dataset('MNIST', config_path='config.json') # only MNIST
model_test = EfficientCapsNet('MNIST', mode='test', verbose=False)
model_test.load_graph_weights()
model_play = EfficientCapsNet('MNIST', mode='play', verbose=False)
model_play.load_graph_weights()
model_test.evaluate(mnist_dataset.X_test, mnist_dataset.y_test)
AffineVisualizer(model_play, mnist_dataset.X_test, mnist_dataset.y_test, hist=True).start()


import numpy as np
os.chdir('/content/Testimg')

for filename in os.listdir('./'):
    print(filename)
    img = tf.keras.utils.load_img(filename, color_mode="grayscale")
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])/255.
    pred = model_test.predict(input_arr)
    print(np.argmax((max(pred[0]))))
    print('\n\n')


import numpy as np
import tensorflow as tf
from utils.layers import PrimaryCaps, FCCaps, Length, Mask


def efficient_capsnet_graph(input_shape):
    """
    Efficient-CapsNet graph architecture.

    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(32,5,activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128,3,2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
    x = tf.keras.layers.BatchNormalization()(x)
    x = PrimaryCaps(128, 9, 16, 8)(x)
    
    digit_caps = FCCaps(10,16)(x)
    
    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs,outputs=[digit_caps, digit_caps_len], name='Efficient_CapsNet')

model = efficient_capsnet_graph((1, 28,28, 1))
tf.keras.models.save_model(model, './')

converter = tf.lite.TFLiteConverter.from_saved_model('/content/Efficient-CapsNet')
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)



#161,824 params

# 7.png
# 7

# 3.png
# 3

# 1.png
# 1

# 6.png
# 6

# 9.png
# 9

# 5.png
# 5

# 0.png
# 0

# 4.png
# 4

# 8.png
# 8

# 2.png
# 2