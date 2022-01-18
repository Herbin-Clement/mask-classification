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
