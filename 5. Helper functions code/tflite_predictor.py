import tensorflow as tf
import numpy as np
import os

# print(os.getcwd())
#loaded = tf.keras.models.load_model('./')
# #loaded_model = tf.keras.models.load_model('saved_model')

# for filename in os.listdir('./Testimg'):
#     print(filename)
#     img = tf.keras.utils.load_img('./Testimg/'+filename, color_mode="grayscale")
#     input_arr = tf.keras.preprocessing.image.img_to_array(img)
#     input_arr = np.array([input_arr])/255.
#     pred = loaded.predict(input_arr)
#     print(np.argmax(pred))
#     print('\n\n')


interpreter = tf.lite.Interpreter(model_path="Own_model/ownmodel7.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']



for filename in os.listdir('./Testimg'):
    print(filename)
    img = tf.keras.utils.load_img('./Testimg/'+filename, color_mode="grayscale")
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])/255.
    interpreter.set_tensor(input_details[0]['index'], input_arr)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(np.argmax(output_data[0]))
    print("\n")
    # 'output' is dictionary with all outputs from the inference.
    # In this case we have single output 'result'.
