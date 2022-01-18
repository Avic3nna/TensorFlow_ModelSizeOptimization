import tensorflow as tf
import os

print(os.getcwd())

model = tf.keras.models.load_model('./')

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_dtype=False,
    show_layer_names=False, rankdir='TB', expand_nested=False, dpi=600,
    layer_range=None
)