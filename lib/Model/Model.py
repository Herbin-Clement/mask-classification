from genericpath import isdir
from logging import root
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

class Model:

    def __init__(self, root_dir, dataset_dir, nb_epochs=10, batch_size=32):
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train") 
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.validation_dir = os.path.join(self.dataset_dir, "validation")
        self.weights_dir = os.path.join(self.root_dir, "weights", type(self).__name__)
        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)
        self.cur_save_id = len(list(os.walk(self.weights_dir))[0][1])
        self.last_save_id = len(list(os.walk(self.weights_dir))[0][1]) - 1
        if self.last_save_id != -1:
            self.last_checkpoint_path = os.path.join(self.weights_dir, f"{self.last_save_id}")
        self.checkpoint_path = os.path.join(self.weights_dir, f"{self.cur_save_id}", "cp-{epoch:04d}.ckpt")
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

    def create_compile_model(self):
        """
        create and compile the model with keras
        """
        pass

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
        class_mode="categorical" 
        )

        self.test_data = test_datagen.flow_from_directory(
        self.test_dir,
        target_size=(224, 224),
        batch_size=self.batch_size,
        class_mode="categorical" 
        )
    
    def fit_model(self):
        """
        fit the model
        """
        self.__load_images()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq="epoch"
        )
        self.history = self.model.fit(self.train_data,
                    epochs=self.nb_epochs, 
                    steps_per_epoch=len(self.train_data), 
                    validation_data=self.test_data, 
                    validation_steps=len(self.test_data),
                    callbacks=[cp_callback]
        )

    def load_weights(self):
        """
        load the last model weights
        """
        if self.last_save_id == -1:
            print("No model to load")
            exit()
        self.model.load_weights(os.path.join(self.last_checkpoint_path, "cp-0005.ckpt"))

    def get_model(self):
        """
        return the tensorflow model
        """
        return self.model

    def get_history(self):
        """
        get the model history
        """
        if hasattr(self, "history"):
            return self.history
        else:
            print("no history save !")