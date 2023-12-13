import numpy as np
import tensorflow as tf

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_and_preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32, 3))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape((1, 32, 32, 3))

    return image_array


def predict_class(image_path, model):
    image = load_and_preprocess_image(image_path)

    predictions = model.predict(image)

    predicted_class = np.argmax(predictions, axis=-1)

    return predicted_class


if __name__ == "__main__":
    loaded_model = tf.keras.models.load_model('../model/cifar-10_test_model.keras')

    new_image_path = ('0001.jpg')

    predicted_class = predict_class(new_image_path, loaded_model)

    print("Predicted Class: %s" % class_names[predicted_class[0]])
