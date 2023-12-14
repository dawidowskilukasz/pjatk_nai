"""
Fashion MNIST Image Classification using TensorFlow Neural Networks

This script performs image classification on the Fashion MNIST dataset using two neural network models implemented in
TensorFlow.

Ensure the '3_fashion_mnist_samples/' directory exists for saving image samples, and the model files
'3_fashion_mnist_model_1.keras' and '3_fashion_mnist_model_2.keras' are present for model loading. The Fashion MNIST
dataset is expected to be available through the TensorFlow dataset module.
"""

import random
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf

# https://github.com/zalandoresearch/fashion-mnist
SAMPLES_PATH = "3_fashion_mnist_samples/"
MODEL_NAME_1 = "3_fashion_mnist_model_1.keras"
MODEL_NAME_2 = "3_fashion_mnist_model_2.keras"

class_names = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

"""
Loading the dataset
"""
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

"""
Generate some sample data
"""
samples_train = random.sample(range(0, 60000), 5)
samples_test = random.sample(range(0, 10000), 5)


def generate_samples(samples, X, y):
    for s in samples:
        image = X[s]
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, aspect='auto', cmap='gray')
        path = SAMPLES_PATH + str(s) + "_" + str(y[s])
        plt.savefig(path, bbox_inches='tight')


generate_samples(samples_train, X_train, y_train)
generate_samples(samples_test, X_test, y_test)

"""
Creating and/or loading neural network model 1 (3 layers)
"""
try:
    model_1 = tf.keras.models.load_model(MODEL_NAME_1)
except:
    model_1 = None
if not model_1:
    n_epochs = 50

    model_1 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name="X"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model_1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model_1.summary()
    model_1.fit(X_train, y_train, epochs=n_epochs)
    model_1.save(MODEL_NAME_1)

"""
Creating predictions for model 1 (3 layers)
"""
train_accuracy_model_1 = model_1.evaluate(X_train, y_train)[1]
test_accuracy_model_1 = model_1.evaluate(X_test, y_test)[1]

print(f"Train Classification Accuracy (Model 1: 3 Layers): {train_accuracy_model_1:.4f}")
print(f"Test Classification Accuracy (Model 1: 3 Layers): {test_accuracy_model_1:.4f}")


"""
Creating and/or loading neural network model 2 (1 layers)
"""
try:
    model_2 = tf.keras.models.load_model(MODEL_NAME_2)
except:
    model_2 = None
if not model_2:
    n_epochs = 50

    model_2 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name="X"),
        tf.keras.layers.Dense(384, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model_2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model_2.summary()
    model_2.fit(X_train, y_train, epochs=n_epochs)
    model_2.save(MODEL_NAME_2)


"""
Creating predictions for model 2 (1 layer)
"""
train_accuracy_model_2 = model_2.evaluate(X_train, y_train)[1]
test_accuracy_model_2 = model_2.evaluate(X_test, y_test)[1]

print(f"Train Classification Accuracy (Model 2: 1 Layer): {train_accuracy_model_2:.4f}")
print(f"Test Classification Accuracy (Model 2: 1 Layer): {test_accuracy_model_2:.4f}")


def generate_predictions(models, samples):
    """
        Function to generate predictions on user-specified image samples for both models
    """
    print("\n* * * Validation of the Neural Network on Samples * * *\n")
    for s in samples:
        image_path = SAMPLES_PATH + s
        image = tf.keras.utils.load_img(image_path)
        image = tf.image.resize(image, [28, 28])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.copy() / 255.0
        image = image.reshape((28, 28))
        image = np.expand_dims(image, axis=0)

        print(f"\nImage: {s}")
        for i, m in enumerate(models):
            prediction = m.predict(image)
            print(f"Predicted label (Model {i + 1}): {class_names[prediction[0].argmax()]}")
        print(f"Actual label: {class_names[int(s[-5])]}\n")


samples_to_predict = os.listdir(SAMPLES_PATH)
generate_predictions([model_1, model_2], samples_to_predict)
