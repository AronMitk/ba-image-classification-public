from datetime import datetime

import numpy as np
import tensorflow as tf
import json
import logging

from files_filtering import filter_dirs, remove_temp_dir
from training import train

def get_arguments(path):
    file = open(path, mode='r')
    x = file.read()
    file.close()
    return json.loads(x)



PATH = "/Users/arnasmitkevicius/Documents/_datasets/READY_DATASET_TESTING"


def prepare_train_dataset(path, batch_size, image_size):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.9,
        subset="training",
        seed=123,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size)


def prepare_validation_dataset(path, batch_size, image_size):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.9,
        subset="validation",
        seed=123,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size)


def main():
    #get training data
    parameters = get_arguments('/Users/arnasmitkevicius/PycharmProjects/image-classification/main/input-main.json')

    batch_size = parameters["batch_size"]
    img_height = parameters["img_height"]
    img_width = parameters["img_width"]

    trainings = parameters["trainings"]

    print(len(trainings))

    trainings = [t for t in trainings if t["to_train"] == True]

    #train only which are enabled

    for i in trainings:
        try:
            path = filter_dirs(PATH, i["use_labels"])
            validation_dataset = prepare_validation_dataset(path, 156, (img_height, img_width))

            test_dataset = validation_dataset

            class_names = test_dataset.class_names

            print(class_names)
            model = tf.keras.models.load_model(
                '/Users/arnasmitkevicius/PycharmProjects/image-classification/main/saved_model/ALL/v1')

            image_batch, label_batch = test_dataset.as_numpy_iterator().next()
            predictions = model.predict_on_batch(image_batch).flatten()

            print(predictions)
            print(len(predictions))
            print(len(class_names))
            a = np.array(predictions)
            a = a.reshape(-1, len(class_names))
            print(a)
            a = tf.nn.softmax(a)
            print(a)

            # Apply a sigmoid since our model returns logits
            # predictions = tf.nn.sigmoid(predictions)
            # predictions = tf.where(predictions < 0.5, 0, 1)
            predictions = np.argmax(a, axis=-1)

            print('Predictions:\n', predictions)
            print('Labels:\n', label_batch)

            from mlxtend.plotting import plot_confusion_matrix
            from sklearn.metrics import confusion_matrix

            print(type(label_batch))
            print(type(predictions))

            mat = confusion_matrix(label_batch, predictions)
            fig, ax = plot_confusion_matrix(conf_mat=mat, class_names=class_names, figsize=(10,10), show_normed=True)
            fig.show()

            #https: // www.youtube.com / watch?v = TtIjAiSojFE
            from sklearn.metrics import accuracy_score
            print("Accuracy score: ", accuracy_score(label_batch, predictions))

            from sklearn.metrics import classification_report
            print("Classification report:")
            print(classification_report(label_batch, predictions))



        except Exception as e:
            print(e)
        finally:
            remove_temp_dir(PATH)


if __name__ == "__main__":
    main()





