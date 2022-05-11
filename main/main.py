# https://www.tensorflow.org/tutorials/images/transfer_learning

from datetime import datetime

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


def create_logging_file():
    log_file = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename='logs/' + log_file + '.log', format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)


PATH = "/Users/arnasmitkevicius/Documents/_datasets/READY_DATASET_TESTING"


def prepare_train_dataset(path, batch_size, image_size):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size)


def prepare_validation_dataset(path, batch_size, image_size):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size)


def main():
    # create_logging_file()

    #get training data
    parameters = get_arguments('/Users/arnasmitkevicius/PycharmProjects/image-classification/main/input-main.json')
    # logging.info('Parameters: ' + str(parameters))

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
            train_dataset = prepare_train_dataset(path, batch_size, (img_height, img_width))
            validation_dataset = prepare_validation_dataset(path, batch_size, (img_height, img_width))

            sepa = tf.data.experimental.cardinality(validation_dataset).numpy()

            if sepa >= 5:
                sepa = 5

            val_batches = tf.data.experimental.cardinality(validation_dataset)
            test_dataset = validation_dataset.take(val_batches // sepa)
            validation_dataset = validation_dataset.skip(val_batches // sepa)

            print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
            print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

            class_names = train_dataset.class_names

            print(class_names)

            train(i["area"], train_dataset, validation_dataset, test_dataset, class_names, img_height, img_width)
            remove_temp_dir(PATH)
        except Exception as e:
            print(e)
        finally:
            remove_temp_dir(PATH)


if __name__ == "__main__":
    main()
