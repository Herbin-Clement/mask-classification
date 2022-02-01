from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation

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