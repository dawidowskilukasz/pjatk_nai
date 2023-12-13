import random
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf

# https://github.com/zalandoresearch/fashion-mnist
SAMPLES_PATH = "3_fashion_mnist_samples/"
MODEL_NAME = "3_fashion_mnist_model.keras"

class_names = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

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

try:
    model = tf.keras.models.load_model(MODEL_NAME)
except:
    model = None
if not model:
    d = len(X_train[0])
    n_hidden = 100
    l_rate = .001
    n_epochs = 50

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name="X"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="hard_sigmoid"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(X_train, y_train, epochs=n_epochs)
    model.save(MODEL_NAME)

train_accuracy = model.evaluate(X_train, y_train)[1]
test_accuracy = model.evaluate(X_test, y_test)[1]

print(f"Train Classification Accuracy: {train_accuracy:.4f}")
print(f"Test Classification Accuracy: {test_accuracy:.4f}")


def generate_predictions(model, samples):
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

        prediction = model.predict(image)
        print(f"\nImage: {s}")
        print(f"Predicted label: {class_names[prediction[0].argmax()]}")
        print(f"Actual label: {class_names[int(s[-5])]}\n")


samples_to_predict = os.listdir(SAMPLES_PATH)
generate_predictions(model, samples_to_predict)
