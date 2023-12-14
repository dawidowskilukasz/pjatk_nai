import pandas as pd
import tensorflow as tf
import numpy as np
import random

DATA_FILE = "4_binary_table.csv"
T_SIZE = 0.33
MODEL_NAME = "4_tf_classification.keras"
# 2 ^ 100 = 1 267 650 600 228 229 401 496 703 205 376
DATA_SIZE = 100
FRACTION = 10

class_names = ['X>Y', 'X=Y', 'X<Y']


def generate_data():
    rows_A = []
    rows_B = []
    results = []
    for a in range(2 ** int(DATA_SIZE / FRACTION)):
        binary_a = format(a, '0' + str(DATA_SIZE) + 'b')
        for b in range(2 ** int(DATA_SIZE / FRACTION)):
            binary_b = format(b, '0' + str(DATA_SIZE) + 'b')
            a_b_greater = 1 if a > b else 0
            a_b_equal = 1 if a == b else 0
            a_b_less = 1 if a < b else 0
            rows_A.append(list(map(int, binary_a)))
            rows_B.append(list(map(int, binary_b)))
            results.append([a_b_greater, a_b_equal, a_b_less])

    columns = list(reversed(range(DATA_SIZE)))
    columns_A = ["A" + str(c) for c in columns]
    columns_B = ["B" + str(c) for c in columns]

    df = pd.concat(
        [pd.DataFrame(rows_A, columns=columns_A), pd.DataFrame(rows_B, columns=columns_B),
         pd.DataFrame(results, columns=class_names)], axis=1)
    df.to_csv(DATA_FILE, index=False)
    return df


try:
    data = pd.read_csv(DATA_FILE)
except:
    data = None
if data is None:
    data = generate_data()

X_A = np.array(data.loc[:, data.columns.str.startswith('A')])
X_B = np.array(data.loc[:, data.columns.str.startswith('B')])
y = np.array(data[class_names])

try:
    model = tf.keras.models.load_model(MODEL_NAME)
except:
    model = None
if not model:
    d = len(X_A[0])
    n_epochs = 25
    l_rate = .001
    n_hidden = 128
    b_size = 20000
    dropout = 0.5

    input_A = tf.keras.layers.Input(shape=(d,), name='input_A')
    input_B = tf.keras.layers.Input(shape=(d,), name='input_B')
    input_merged = tf.keras.layers.Concatenate(axis=1)([input_A, input_B])
    hidden_layer = tf.keras.layers.Dense(n_hidden, input_dim=2, activation='relu', name='hidden_layer')(input_merged)
    dropout1 = tf.keras.layers.Dropout(rate=dropout)(hidden_layer)
    hidden_layer2 = tf.keras.layers.Dense(n_hidden * 2, activation='relu', name='hidden_layer2')(dropout1)
    dropout2 = tf.keras.layers.Dropout(rate=dropout)(hidden_layer2)
    hidden_layer3 = tf.keras.layers.Dense(n_hidden, activation='relu', name='hidden_layer3')(dropout2)
    dropout3 = tf.keras.layers.Dropout(rate=dropout)(hidden_layer3)
    output_layer = tf.keras.layers.Dense(3, activation='softmax', name='output')(dropout3)

    model = tf.keras.Model(inputs=[input_A, input_B], outputs=output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    model.fit([X_A, X_B], y, epochs=n_epochs, batch_size=b_size)
    model.save(MODEL_NAME)


accuracy = model.evaluate([X_A, X_B], y)[1]
print(f"Classification Accuracy: {accuracy:.4f}")

def generate_predictions(model, samples):
    print("\n* * * Validation of the Neural Network on Samples * * *\n")
    for _ in range(samples):
        a = random.randint(1000000000000, 2 ** DATA_SIZE)
        b = random.randint(1000000000000, 2 ** DATA_SIZE)
        a_bin = np.expand_dims(np.array(list(map(int, format(a, '0' + str(DATA_SIZE) + 'b')))), axis=0)
        b_bin = np.expand_dims(np.array(list(map(int, format(b, '0' + str(DATA_SIZE) + 'b')))), axis=0)

        print(f"X: {a}")
        print(f"Y: {b}")
        print("Actual Comparison: ", end="")
        if a > b:
            print("X>Y")
        elif a == b:
            print("X=Y")
        else:
            print("X<Y")

        predictions = model.predict([a_bin, b_bin])
        predicted_class = np.argmax(predictions, axis=-1)
        print(f"Predicted Comparison: {class_names[predicted_class[0]]}\n")


generate_predictions(model, 10)
