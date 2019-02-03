from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Activation, Reshape
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
import keras
import numpy as np


class GAN():
    def __init__(self):
        # Size of the grayscale images in the dataset
        self.img_rows = 28
        self.img_cols = 28
        self.num_channels = 1

        # Size of the noise vector used as input to the generator
        self.latent_dim = 100
        self.input_shape = (self.img_rows, self.img_cols, self.num_channels)

        optim = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optim,
                                metrics=['accuracy'])
        self.generator = self.build_generator()



    def build_generator(self):
        model = Sequential()

        model.add(Dense(1024, input_dim=self.latent_dim))
        model.add(Reshape((4, 4, -1)))
        model.add(Conv2DTranspose(64, 4))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, 4))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, 4))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, 4))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Activation('tanh'))
        return model

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(4, 64))
        model.add(LeakyReLU())
        model.add(Conv2D(4, 64))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2D(4, 64))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2D(4, 64))
        model.add(Dense(1, activation="sigmoid"))
        return model

gan = GAN()
