# Import TensorFlow and other necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load the MNIST dataset (handwritten digits)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   
    layers.Dense(128, activation='relu'),   
    layers.Dropout(0.2),                   # Dropout layer to prevent overfitting
    layers.Dense(10, activation='softmax') # Output layer with 10 units for 10 classes (digits 0-9)
])

# Compile the model with loss function, optimizer, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Output the accuracy
print('\nTest accuracy:', test_acc)
