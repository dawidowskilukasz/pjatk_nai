import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RESULT_DECISION_TREE = round(0.9712187841935587, 5)
RESULT_SVM = round(0.9841467436985966, 5)
MODEL_NAME = "1_credit_card_fraud.keras"

# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
credit_card_data = pd.read_csv('1_creditcard_dataset.csv')
credit_card_data.drop([credit_card_data.columns[0]], axis=1, inplace=True)

y = np.array(credit_card_data['Class'])
X = np.array(credit_card_data.drop('Class', axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, stratify=y)

w_train, w_test = y_train, y_test
w_train, w_test = w_train.astype('float64'), w_test.astype('float64')
w_train[w_train == 0], w_test[w_test == 0] = 0.00175861280621845, 0.00175861280621845
w_train[w_train == 1], w_test[w_test == 1] = 1.01626016260163, 1.01626016260163

try:
    model = tf.keras.models.load_model(MODEL_NAME)
except:
    model = None
if not model:
    d = len(X_train[0])
    n_hidden = 100
    l_rate = .001
    n_epochs = 100

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(d,), dtype=tf.float32, name='X'),
        tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden_layer'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=n_epochs)
    model.save(MODEL_NAME)

y_train_pred = tf.round(model.predict(X_train))
y_test_pred = tf.round(model.predict(X_test))

train_weighted_score = accuracy_score(y_train, y_train_pred, sample_weight=w_train)
print("Train Weighted Classification Accuracy: %f" % train_weighted_score)
test_weighted_score = accuracy_score(y_test, y_test_pred, sample_weight=w_test)
print("Test Weighted Classification Accuracy: %f" % test_weighted_score)

print("\n* * * Comparison with Decision Tree and SVM Classifiers * * *\n")
print("Test Weighted Classification Accuracy Using Decision Tree Classifier: %f" % RESULT_DECISION_TREE)
print("Test Weighted Classification Accuracy Using SVM Classifier: %f" % RESULT_SVM)

if RESULT_DECISION_TREE > test_weighted_score:
    print("\nDecision Tree Classifier provided the most accurate result")
elif RESULT_SVM > test_weighted_score:
    print("\nSVM Classifier provided the most accurate result")
else:
    print("\nTensorFlow NeuralNetwork provided the most accurate result")
