from tensorflow.keras.applications.inception_v3 import InceptionV3

from lib.Model import Model

class Tl_InceptionV3(Model):

    def __init__(self, root_dir, dataset_dir, save_dir, batch_size=32):
        Model.__init__(root_dir, dataset_dir, save_dir, batch_size)

    def compile_model(self):
        """
        create and compile the model with keras
        """
        pre_trained_model = InceptionV3(input_shape=(224, 224, 3),
                                include_top = False,
                                weights = 'imagenet')

        for layer in pre_trained_model.layers:
        layer.trainable = False

        x = Flatten()(pre_trained_model.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(4, activation="softmax")(x)

        model = Model(pre_trained_model.input, x)

        model.compile(optimizer=Adam(),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])