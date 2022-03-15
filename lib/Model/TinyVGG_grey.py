from lib.Model.Model import Model

import tensorflow as tf 
from  tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TinyVGG_grey(Model):

    def __init__(self, root_dir, dataset_dir, nb_epochs=10, batch_size=32):
        Model.__init__(self, root_dir, dataset_dir, nb_epochs, batch_size)
        self.compile_model()

    def compile_model(self):
        """
        create and compile the model with keras
        """
        self.model = Sequential([
                    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 1)),
                    Conv2D(10, 3, activation="relu"),
                    MaxPool2D(pool_size=2),
                    Conv2D(10, 3, activation="relu"),
                    Conv2D(10, 3, activation="relu"),
                    MaxPool2D(pool_size=2),
                    Flatten(),
                    Dense(4, activation="softmax")
        ])
        self.model.compile(loss="categorical_crossentropy",
                    optimizer=Adam(),
                    metrics=["accuracy"])

    def __load_images(self):
        """
        load images with ImageDataGenerator,and pre-process them with normalization, rescale, shift, zoom and rotation
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            height_shift_range=0.1,
            width_shift_range=0.1,
            zoom_range=0.1,
            rotation_range=0.1
        )

        test_datagen = ImageDataGenerator(
            rescale=1./255,
        )

        self.train_data = train_datagen.flow_from_directory(
        self.train_dir,
        target_size=(224, 224),
        batch_size=self.batch_size,
        class_mode="categorical",
        color_mode="grayscale"
        )

        self.test_data = test_datagen.flow_from_directory(
        self.test_dir,
        target_size=(224, 224),
        batch_size=self.batch_size,
        class_mode="categorical" 
        )