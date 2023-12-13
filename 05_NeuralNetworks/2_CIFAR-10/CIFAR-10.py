import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Flatten, Dense

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, x_test = x_train.reshape((len(x_train), 32, 32, 3)), x_test.reshape((len(x_test), 32, 32, 3))
y_train, y_test = y_train.flatten(), y_test.flatten()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=0)

n_hidden = 100
l_rate = 0.001
n_epochs = 50

input_layer = Input(shape=(32, 32, 3), name='input_layer')
flatten_layer = Flatten()(input_layer)
hidden_layer = Dense(n_hidden, activation='relu', name='hidden_layer')(flatten_layer)
hidden_layer2 = Dense(n_hidden, activation='relu', name='hidden_layer2')(hidden_layer)
hidden_layer3 = Dense(n_hidden, activation='relu', name='hidden_layer3')(hidden_layer2)
output_layer = Dense(10, activation='softmax', name='output')(hidden_layer3)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=n_epochs)

y_train_pred = np.argmax(model.predict(X_train), axis=-1)
y_test_pred = np.argmax(model.predict(X_test), axis=-1)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Classification Accuracy: {train_accuracy:.4f}")
print(f"Test Classification Accuracy: {test_accuracy:.4f}")

model.save('cifar-10_test_model.keras')

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
plt.show()
