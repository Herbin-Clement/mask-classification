from Model import Model

class TinyVGG(Model):

    def __init__(self, root_dir, dataset_dir, save_dir, batch_size=32):
        Model.__init__(root_dir, dataset_dir, save_dir, batch_size)

    def compile_model(self):
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