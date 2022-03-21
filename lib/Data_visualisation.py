from random import randint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
import itertools
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image


def displayRandomImage(df, image_directory):
    """
    Display a ramdom image from the DataFrame with it ID and it type

    :param df:
    :image_directory: the path of the image directory
    """
    random_index = randint(0, len(df))
    image = df.iloc[random_index]
    img_info = (image.ID, image.TYPE, image['name'])
    disp = mpimg.imread(image_directory + '/'+img_info[2])
    plt.axis('off')
    plt.imshow(disp)
    plt.title('ID : ' + str(img_info[0]) + '  Type : ' + str(img_info[1]))


def print_loss_accuracy(history):
    """
    plot loss and accuracy data of a model which is train
    :param history: dict
    """
    loss = history["loss"]
    val_loss = history["val_loss"]
    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    epochs = range(len(loss))

    plt.plot(epochs, loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.figure()

    plt.plot(epochs, accuracy, label="Train accuracy")
    plt.plot(epochs, val_accuracy, label="Test accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def predict_validation_image(path_directory, model, verbose=True):
    """
    predict the class of an image
    :param path_directory: str
    :param model: model 
    """
    img = tf.keras.utils.load_img(path_directory,  target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = np.argmax(model.predict(img_array)) + 1
    split_path = os.path.split(path_directory)
    label = split_path[len(split_path) - 1].split("_")[1]
    if verbose:
        print(
            f"The model predict class {pred} and the class is {label} for {path_directory}")
    return label, pred


def predict_image(image, model, verbose=True):
    image = cv2.resize(image, (224, 224))
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    b, g, r = tf.unstack(tensor, axis=-1)
    tensor = tf.stack([r, g, b], axis=-1)
    tensor = tf.expand_dims(tensor, 0)
    pred = np.argmax(model.predict(tensor)) + 1
    if verbose:
        print(f"The model predict class {pred}")
    return pred

def confusion_matrix(validation_folder, model, y_true):
      images = []
      for img in os.listdir(validation_folder):
            img = os.path.join(validation_folder, img)
            img = image.load_img(img, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images.append(img)

      images = np.vstack(images)
      classes = model.predict(images)
      print(classes)

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / \
        cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    # colors will represent how 'correct' a class is, darker == better
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           # create enough axis slots for each class
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           # axes will labeled with class names (if they exist) or ints
           xticklabels=labels,
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
