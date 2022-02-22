from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import os

class Model:

    def __init__(self, root_dir, dataset_dir, save_dir, batch_size=32):
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train") 
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.validation_dir = os.path.join(self.dataset_dir, "validation")
        self.cur_save_id = len(list(os.walk(save_dir))[0][1])
        self.last_save_id = len(list(os.walk(save_dir))[0][1]) - 1
        self.checkpoint_path = os.path.join(root_dir, "{cur_save_id}", "cp-{epoch:04d}.ckpt")
        if self.last_save_id != -1:
            self.last_checkpoint_path = os.path.join(root_dir, "{last_save_id}", "cp-{epoch:04d}.ckpt")
        self.batch_size = batch_size
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

    def __get_images(self):
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
        self.__get_images()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq="epoch"
        )
        self.history = self.model.fit(self.train_data,
                    epochs=30, 
                    steps_per_epoch=len(self.train_data), 
                    validation_data=self.test_data, 
                    validation_steps=len(self.test_data),
                    callbacks=[cp_callback]
        )

    def load_weights(self):
        if self.last_save_id == -1:
            print("No model to load")
            exit()
        self.model.load_weights(os.path.join(self.last_checkpoint_path, "cp-0030.ckpt"))

    def get_model(self):
        return self.model