from random import randint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import os
import cv2
from keras.preprocessing.image import save_img

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
    plt.title('ID : ' + str(img_info[0]) + '  Type : '+ str(img_info[1]))

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
  print("pred_load_jpg")
  print(img_array)
  pred = np.argmax(model.predict(img_array)) + 1
  split_path = os.path.split(path_directory)
  label = split_path[len(split_path) - 1].split("_")[1]
  if verbose:
    print(f"The model predict class {pred} and the class is {label} for {path_directory}")
  return label, pred

def predict_image(image, model, image_name, image_name2, verbose=True):
  cv2.imwrite(image_name, image)
  image = cv2.resize(image, (224, 224))
  # cv2.imwrite(image_name, image)
  # image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
  tensor = tf.convert_to_tensor(image, dtype=tf.float32)
  b, g, r = tf.unstack(tensor, axis=-1)
  tensor = tf.stack([r, g, b], axis=-1)
  tensor = tf.expand_dims(tensor, 0)
  save_img(image_name2, tensor[0])
  print("pred_no_load_img")
  print(tensor)
  pred = np.argmax(model.predict(tensor)) + 1
  if verbose:
    print(f"The model predict class {pred}")
  return pred