from random import randint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


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