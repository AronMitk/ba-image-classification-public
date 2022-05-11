import os

import numpy as np
import tensorflow as tf

from charts_helper import draw_predictions, draw_accuracy_data, draw_accuracy_loss_fine_data, draw_augmentation, \
    draw_pretrained_data, draw_confusion_matrix
from model_utils import save_model

_CHARTS_PATH = '/Users/arnasmitkevicius/PycharmProjects/image-classification'


def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(-.5, -.2),
        tf.keras.layers.RandomContrast(0.2),
    ])


def train(area, train_dataset, validation_dataset, test_dataset, class_names, IMG_HEIGHT, IMG_WIDTH):
    global _CHARTS_PATH

    #create charts folder
    CHARTS_PATH = os.path.join(_CHARTS_PATH, 'charts', area)
    os.makedirs(CHARTS_PATH, exist_ok=True)

    #draw pretrained data
    draw_pretrained_data(train_dataset, class_names, CHARTS_PATH)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    data_augmentation = get_augmentation()

    draw_augmentation(train_dataset, data_augmentation, CHARTS_PATH)

    preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

    tf.keras.layers.Rescaling(1. / 127.5, offset=-1)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH) + (3,)
    base_model = tf.keras.applications.EfficientNetB3(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False

    print(base_model.summary())

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    def getModel():
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        # x = global_average_layer(x)
        # x = tf.keras.layers.Dropout(0.2)(x)

        x = global_average_layer(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1536, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(units=1536, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)


        outputs = prediction_layer(x)
        return tf.keras.Model(inputs, outputs)

    model = getModel()

    base_learning_rate = 0.0001

    STEPS_PER_EPOCH = tf.data.experimental.cardinality(train_dataset).numpy() // 8
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        0.00001,
        decay_steps=STEPS_PER_EPOCH * 100
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    print(model.summary())

    len(model.trainable_variables)

    initial_epochs = 10

    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(initial_epochs)

    draw_accuracy_data(epochs_range, acc, val_acc, loss, val_loss, CHARTS_PATH)

    base_model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 100),
                  metrics=['accuracy'])

    print(model.summary())

    print(len(model.trainable_variables))

    fine_tune_epochs = 2
    total_epochs = initial_epochs + fine_tune_epochs

    checkpoint_path = area + "_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(os.path.join('training', area, checkpoint_path))

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                     save_weights_only=True,
                                                     verbose=1)

    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset,
                             callbacks=[cp_callback])

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    draw_accuracy_loss_fine_data(initial_epochs, acc, val_acc, loss, val_loss, CHARTS_PATH)

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    save_model(model, 'saved_model/' + area + '/v1')

    # TESTING
    # Retrieve a batch of images from the test set
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

    draw_predictions(image_batch, class_names, predictions, CHARTS_PATH)

    from mlxtend.plotting import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix

    print(type(label_batch))
    print(type(predictions))

    mat = confusion_matrix(label_batch, predictions)
    draw_confusion_matrix(mat, class_names, CHARTS_PATH)