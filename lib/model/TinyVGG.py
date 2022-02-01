from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import Mean

class TinyVGG(Model):
    def __init__(self):
        super(TinyVGG, self).__init__()
        self.conv_1_1 = Conv2D(10, 3, input_shape=(224, 224, 3))
        self.relu_1_1 = Activation('relu')
        self.conv_1_2 = Conv2D(10, 3)
        self.relu_1_2 = Activation('relu')
        self.max_pool_1 = MaxPool2D((2, 2))

        self.conv_2_1 = Conv2D(10, 3)
        self.relu_2_1 = Activation('relu')
        self.conv_2_2 = Conv2D(10, 3)
        self.relu_2_2 = Activation('relu')
        self.max_pool_2 = MaxPool2D((2, 2))

        self.flatten = Flatten()
        self.final = Dense(4, activation='softmax')

    def call(self, x):
        x = self.conv_1_1(x)
        x = self.relu_1_1(x)
        x = self.conv_1_2(x)
        x = self.relu_1_2(x)
        x = self.max_pool_1(x)
        
        x = self.conv_2_1(x)
        x = self.relu_2_1(x)
        x = self.conv_2_2(x)
        x = self.relu_2_2(x)
        x = self.max_pool_2(x)

        x = self.flatten(x)
        return final.fc(x)

model = TinyVGG()
loss_object = CategoricalCrossentropy()
optimizer = Adam()

train_accuracy = CategoricalAccuracy()
test_accuracy = CategoricalAccuracy()

train_mean_loss = Mean()
test_mean_loss = Mean()

@tf.function
def train_step(image_batch, label_batch):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(5):
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)