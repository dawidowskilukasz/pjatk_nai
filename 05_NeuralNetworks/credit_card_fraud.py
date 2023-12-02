import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
credit_card_data = pd.read_csv('creditcard.csv')
credit_card_data.drop([credit_card_data.columns[0]], axis=1, inplace=True)

y = np.array(credit_card_data['Class'])
X = np.array(credit_card_data.drop('Class', axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, stratify=y)

w_train = y_train
w_train = w_train.astype('float64')
w_train[w_train == 0] = 0.000175861280621845
w_train[w_train == 1] = 0.101626016260163

w_test = y_test
w_test = w_test.astype('float64')
w_test[w_test == 0] = 0.000175861280621845
w_test[w_test == 1] = 0.101626016260163

d = len(X_train[0])
n_hidden = 50
l_rate = .001
n_epochs = 10

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(d,), dtype=tf.float32, name='X'),
    tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden_layer'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate), loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=n_epochs)

# https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
# 190820 / 32 = 5964
# 93987 / 32 = 2938
y_train_pred = tf.round(model.predict(X_train))
y_test_pred = tf.round(model.predict(X_test))

train_weighted_score = accuracy_score(y_train, y_train_pred, sample_weight=w_train)
print("Train Weighted Classification Accuracy: %f" % train_weighted_score)
test_weighted_score = accuracy_score(y_test, y_test_pred, sample_weight=w_test)
print("Test Weighted Classification Accuracy: %f" % test_weighted_score)
