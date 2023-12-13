import random
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

SAMPLES_PATH = "./2_fashion_mnist_samples/"
MODEL_NAME = "3_fashion_mnist_model.keras"

fashion_mnist = tf.keras.datasets.fashion_mnist

labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

samples_train = random.sample(range(0, 60000), 5)
samples_test = random.sample(range(0, 10000), 5)


def generate_samples(samples):
    for s in samples:
        image = X_train[s]
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, aspect='auto', cmap='gray')
        path = SAMPLES_PATH + str(s) + "_" + str(y_train[s])
        plt.savefig(path, bbox_inches='tight')


generate_samples(samples_train)
generate_samples(samples_test)

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

print("Train Classification Accuracy: %f" % train_accuracy)
print("Test Classification Accuracy: %f" % test_accuracy)


def generate_predictions(model, samples):
    print("\n* * * Validation of the Neural Network on Samples * * *\n")
    for s in samples:
        img_path = SAMPLES_PATH + s
        img = tf.keras.utils.load_img(img_path)
        img = tf.image.resize(img, [28, 28])
        img = tf.image.rgb_to_grayscale(img)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.copy() / 255.0
        img = img.reshape((28, 28))
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        print("\nImage:", s)
        print("Predicted label:", prediction[0].argmax())
        print("Actual label:", s[-5])


samples_to_predict = os.listdir(SAMPLES_PATH)
generate_predictions(model, samples_to_predict)
