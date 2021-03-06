from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model as kModel

from lib.Model.Model import Model

class TL_VGG19(Model):

    def __init__(self, root_dir, dataset_dir, nb_epochs=10, batch_size=32):
        Model.__init__(self, root_dir, dataset_dir, nb_epochs=nb_epochs, batch_size=batch_size)
        self.compile_model()

    def compile_model(self):
        """
        create and compile the model with keras
        """
        pre_trained_model = VGG19(input_shape=(224, 224, 3),
                                include_top = False,
                                weights = 'imagenet', classes=4)

        for layer in pre_trained_model.layers:
            layer.trainable = False

        x = Flatten()(pre_trained_model.output)
        x = Dense(4, activation="softmax")(x)

        self.model = kModel(pre_trained_model.input, x)

        self.model.compile(optimizer=Adam(),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])