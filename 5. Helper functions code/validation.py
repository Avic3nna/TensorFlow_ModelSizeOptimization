import tensorflow as tf
import numpy as np
import os

# print(os.getcwd())
loaded = tf.keras.models.load_model('./')
# #loaded_model = tf.keras.models.load_model('saved_model')

for filename in os.listdir('./Testimg'):
    print(filename)
    img = tf.keras.utils.load_img('./Testimg/'+filename, color_mode="grayscale")
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])/255.
    pred = loaded.predict(input_arr)
    print(np.argmax(pred))
    print('\n\n')


## convert and save model to tf lite

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('./') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)