import tensorflow as tf
import pathlib

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
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build sequential model by stacking layers, choose optimizer and loss function
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

predictions = model(x_train[:1]).numpy()
predictions_prob = tf.nn.softmax(predictions).numpy()
print ('Probabilities for each class: ' + str(predictions_prob))

# Take a vector of logits and True index and return scalar loss for each example
# This loss is equal to the negative log probability of the true class: It is zero if the model is sure of the correct class.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_initial = loss_fn(y_train[:1], predictions).numpy()
print('Untrained model inital loss: ' + str(loss_initial))

# Train model
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Adjust model parameters to minimize the loss and train it
model.fit(x_train, y_train, epochs=20)

# Evaluate model performance
model.evaluate(x_test, y_test, verbose=2)

tf.keras.models.save_model(model, './')