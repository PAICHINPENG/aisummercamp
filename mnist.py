import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Add a channel dimension for CNN input
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer with 10 units (one for each class)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=2, batch_size=200, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")

import matplotlib.pyplot as plt
import numpy as np
import random

# Function to show images with predicted labels
def show_test_images_predictions(images, labels, predictions, num=8):
    plt.figure(figsize=(12, 6))
    random_indices = random.sample(range(len(images)), num)
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[idx, ..., 0], cmap='binary')
        predicted_label = np.argmax(predictions[idx])
        true_label = labels[idx]
        plt.title(f'Predicted: {predicted_label}, True: {true_label}')
        plt.axis('off')
    plt.show()

# Assuming you have already trained the model and evaluated on the test set
# Get the model predictions on the test set
test_predictions = model.predict(test_images)

# Show 8 random test images along with their predicted labels
show_test_images_predictions(test_images, test_labels, test_predictions, num=8)

from google.colab import drive
drive.mount('/content/drive')

model.save('/content/drive/MyDrive/SummerCamp/')

import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('/content/drive/MyDrive/SummerCamp/gpt.h5')

# Load and preprocess your custom digit image
# Replace 'path_to_your_image' with the actual path of your image
image_path = '/content/drive/MyDrive/SummerCamp/one.jpg'
img = plt.imread(image_path)
img = img.astype('float32') / 255.0
img = tf.image.rgb_to_grayscale(img)  # Convert to grayscale
img = tf.image.resize(img, (28, 28))  # Resize to 28x28 pixels
img = img.numpy()  # Convert to NumPy array
input_image = img[..., tf.newaxis]  # Add a channel dimension
input_image_batch = tf.expand_dims(input_image, axis=0)  # Add a batch dimension

# Make predictions on the custom image
prediction = model.predict(input_image_batch)

# Display the custom image and its predicted label
plt.imshow(img[..., 0], cmap='binary')
predicted_label = tf.argmax(prediction[0]).numpy()
plt.title(f'Predicted Label: {predicted_label}')
plt.axis('off')
plt.show()


