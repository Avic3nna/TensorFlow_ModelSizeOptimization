import tensorflow as tf
import pathlib
import numpy as np
tf.keras.backend.set_image_data_format('channels_last')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print(tf.__version__)
# Export saved model
#export_dir = 'mymodel'

# Load and prepare MNIST dataset
mnist = tf.keras.datasets.mnist

# Normalize dataset
(x_train, y_train) , (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.expand_dims(x_train, axis = -1)
x_test = np.expand_dims(x_test, axis = -1)

x_train = tf.image.resize(x_train, [32,32]) # if we want to resize 
x_test = tf.image.resize(x_test, [32,32]) # if we want to resize 

x_train=np.repeat(x_train, 3, axis = -1)
x_test=np.repeat(x_test, 3, axis = -1)

# x_train = x_train.reshape(-1, 32, 32, 3)
# x_test = x_test.reshape(-1, 32, 32, 3)

# x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
# x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
print(x_train.shape)


model = tf.keras.models.Sequential()
# Build sequential model by stacking layers, choose optimizer and loss function


model.add(tf.keras.applications.vgg16.VGG16(weights='imagenet', input_shape = (32,32,3), include_top=False))
model.add(tf.keras.layers.Conv2D(3, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(32, 32, 3), padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
model.add(tf.keras.layers.Conv2D(4, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(32, 32, 3),padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

for layer in model.layers[:-7]:
    layer.trainable = False
model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Adjust model parameters to minimize the loss and train it
model.fit(x_train, y_train, epochs=20)

# Evaluate model performance
model.evaluate(x_test, y_test, verbose=2)

tf.keras.models.save_model(model, './')