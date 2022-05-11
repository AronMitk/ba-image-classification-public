import os.path

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from mlxtend.plotting import plot_confusion_matrix

def draw_pretrained_data(train_ds, class_names, save_path=None):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            try:
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
            except:
                "error"
            plt.axis("off")

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "pretrained.png"))

    plt.show()


def draw_accuracy_data(epochs_range, acc, val_acc, loss, val_loss, save_path=None):
    print("##############")
    print(acc)
    print(val_acc)
    print(loss)
    print(val_loss)
    print("##############")

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Apmokymo tikslumas')
    plt.plot(epochs_range, val_acc, label='Validacijos tikslumas')
    plt.legend(loc='lower right')
    plt.ylabel('Tikslumas')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Apmokymo ir validacijos tikslumas')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Apmokymo nuostolis')
    plt.plot(epochs_range, val_loss, label='Validacijos nuostolis')
    plt.legend(loc='upper right')
    plt.ylabel('Nuostolis')
    plt.ylim([0, 1])
    plt.title('Apmokymo ir validacijos nuostoliai')
    plt.xlabel('epocha')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "accuracy.png"))

    plt.show()


def draw_accuracy_loss_fine_data(initial_epochs, acc, val_acc, loss, val_loss, save_path=None):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Apmokymo tikslumas')
    plt.plot(val_acc, label='Validacijos tikslumas')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Pradėtas tikslusis derinimas')
    plt.legend(loc='lower right')
    plt.title('Apmokymo ir validacijos tikslumas')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Apmokymo nuostolis')
    plt.plot(val_loss, label='Validacijos nuostolis')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Pradėtas tikslusis derinimas')
    plt.legend(loc='upper right')
    plt.title('Apmokymo ir validacijos nuostoliai')
    plt.xlabel('epocha')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "testing.png"))

    plt.show()


def draw_augmentation(train_dataset, data_augmentation, save_path=None):
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "augmentation.png"))

    plt.show()

def draw_predictions(image_batch, class_names, predictions, save_path=None):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        try:
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].astype("uint8"))
            # plt.title(class_names[predictions[i]])
            plt.title(class_names[predictions[i]])
        except Exception as e:
            print(e)
        plt.axis("off")

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "predictions.png"))

    plt.show()

def draw_confusion_matrix(mat, class_names, save_path=None):
    fig, ax = plot_confusion_matrix(conf_mat=mat, class_names=class_names, figsize=(10, 10), show_normed=True)

    if save_path is not None:
        fig.savefig(os.path.join(save_path, "confusion_matrix.png"))

    fig.show()
