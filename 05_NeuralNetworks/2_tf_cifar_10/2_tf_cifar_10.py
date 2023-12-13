import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import os

SAMPLES_PATH = "2_cifar_10_samples/"
MODEL_NAME = "cifar-10_test_model.keras"

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

X_train, X_test = X_train.reshape((len(X_train), 32, 32, 3)), X_test.reshape((len(X_test), 32, 32, 3))
y_train, y_test = y_train.flatten(), y_test.flatten()

samples_train = random.sample(range(0, 50000), 5)
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
    n_hidden = 100
    l_rate = 0.001
    n_epochs = 50

    input_layer = tf.keras.layers.Input(shape=(32, 32, 3), name='input_layer')
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    hidden_layer = tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden_layer')(flatten_layer)
    hidden_layer2 = tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden_layer2')(hidden_layer)
    hidden_layer3 = tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden_layer3')(hidden_layer2)
    output_layer = tf.keras.layers.Dense(10, activation='softmax', name='output')(hidden_layer3)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=n_epochs)
    model.save(MODEL_NAME)

y_train_pred = np.argmax(model.predict(X_train), axis=-1)
y_test_pred = np.argmax(model.predict(X_test), axis=-1)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Classification Accuracy: {train_accuracy:.4f}")
print(f"Test Classification Accuracy: {test_accuracy:.4f}")

conf_matrix_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_test, interpolation='nearest', cmap='Blues')

for i in range(conf_matrix_test.shape[0]):
    for j in range(conf_matrix_test.shape[1]):
        plt.text(j, i, str(conf_matrix_test[i, j]), ha='center', va='center', color='black')

plt.title('Confusion Matrix Test Set')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('3_tf_cifar-10_conf_matrix.png')


def generate_predictions(model, samples):
    print("\n* * * Validation of the Neural Network on Samples * * *\n")
    for s in samples:
        image_path = SAMPLES_PATH + s
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32, 3))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = image_array / 255.0
        image_array = image_array.reshape((1, 32, 32, 3))

        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=-1)

        print(f"\nImage: {s}")
        print("Predicted Class: %s" % class_names[predicted_class[0]])
        print(f"Actual Class: {class_names[int(s[-5])]}\n")


samples_to_predict = os.listdir(SAMPLES_PATH)
generate_predictions(model, samples_to_predict)
