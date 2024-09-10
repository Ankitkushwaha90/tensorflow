- Here’s a collection of TensorFlow code examples that cover various key aspects of the framework, from basic concepts to more advanced architectures. Each section represents a step in the TensorFlow learning roadmap:

### 1. Basic Tensor Operations
```python
import tensorflow as tf

# Basic tensor operations
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Element-wise addition
add_result = tf.add(a, b)
print("Addition:", add_result)

# Element-wise multiplication
mul_result = tf.multiply(a, b)
print("Multiplication:", mul_result)


# Matrix multiplication
matmul_result = tf.matmul(a, b)
print("Matrix Multiplication:", matmul_result)
```
### 2. Simple Linear Regression
```python
import numpy as np
import tensorflow as tf

# Generate random data
X = np.random.rand(100).astype(np.float32)
Y = 3.5 * X + 2.0

# Create TensorFlow variables for weights and biases
W = tf.Variable([0.], dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)

# Define the linear model
def linear_model(x):
    return W * x + b

# Define loss function (Mean Squared Error)
def loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Define optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.1)

# Training loop
for step in range(200):
    with tf.GradientTape() as tape:
        pred = linear_model(X)
        current_loss = loss(pred, Y)
    gradients = tape.gradient(current_loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if step % 20 == 0:
        print(f"Step {step}, Loss: {current_loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}")
```
### 3. Neural Network for Classification (Using Keras API)
```python 
import tensorflow as tf
from tensorflow.keras import layers

# Load dataset (MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)
```
### 4. Convolutional Neural Network (CNN) for Image Classification
```python
import tensorflow as tf
from tensorflow.keras import layers

# Load CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
model.evaluate(x_test, y_test)
```
### 5. Recurrent Neural Network (RNN) for Text Classification
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

# Load the IMDb dataset for sentiment analysis
max_features = 10000  # Vocabulary size
maxlen = 100  # Sequence length to pad

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define the RNN model
model = tf.keras.Sequential([
    layers.Embedding(max_features, 128, input_length=maxlen),
    layers.SimpleRNN(128, return_sequences=True),
    layers.SimpleRNN(128),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
model.evaluate(x_test, y_test)
```
### 6. Transfer Learning with Pre-trained Model (ResNet)
```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# Load pre-trained ResNet50 without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model
base_model.trainable = False

# Add new layers for classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Example data loading (use real datasets in practice)
# x_train, y_train = ... 
# x_test, y_test = ...

# Train the model
# model.fit(x_train, y_train, epochs=5)

# Evaluate the model
# model.evaluate(x_test, y_test)
```
### 7. Model Deployment with TensorFlow Serving
```bash
# Step-by-step to export model for serving:
# Save a trained model
model.save('saved_model/my_model')

# Exporting model for TensorFlow Serving
saved_model_cli show --dir saved_model/my_model --tag_set serve --signature_def serving_default

# In practice, set up TensorFlow Serving Docker or local instance and deploy.
```
### 8. Using TensorFlow Lite for Mobile Deployment
```python
import tensorflow as tf

# Convert model to TensorFlow Lite format
model = tf.keras.models.load_model('saved_model/my_model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Deploy this model on mobile devices using TensorFlow Lite interpreter
```
These code snippets are comprehensive examples that reflect different stages of TensorFlow learning. They cover everything from basic operations and model building to more advanced architectures like CNNs and RNNs, as well as deployment with TensorFlow Lite and Serving.
