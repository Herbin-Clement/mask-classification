from lib.Model.Model import Model

import tensorflow as tf 
from  tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD

class TinyVGG(Model):

    def __init__(self, root_dir, dataset_dir, nb_epochs=10, batch_size=32):
        Model.__init__(self, root_dir, dataset_dir, nb_epochs=nb_epochs, batch_size=batch_size)
        self.compile_model()

    def compile_model(self):
        """
        create and compile the model with keras
        """
        self.model = Sequential([
                    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
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