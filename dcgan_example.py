from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Activation, Reshape
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import imageio
import os



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


        self.combined = Sequential()
        self.combined.add(self.generator)
        self.discriminator.trainable = False
        self.combined.add(self.discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optim)






    def build_generator(self):
        model = Sequential()

        model.add(Dense(1024, input_dim=self.latent_dim))
        model.add(Reshape((4, 4, -1)))
        model.add(Conv2DTranspose(64, 7, data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, 7, data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, 7, data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(self.num_channels, 7, data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Activation('tanh'))
        # print("Starting Generator layers")
        # for layer in model.layers:
        #     print(layer.input_shape, layer.output_shape)
        # print("Ending Generator layers")
        return model

    def build_discriminator(self):
        model = Sequential()


        model.add(Conv2D(64, 4,input_shape=self.input_shape, data_format="channels_last"))
        model.add(LeakyReLU())
        model.add(Conv2D(64, 4, data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2D(64, 4, data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2D(64, 4, data_format="channels_last"))
        model.add(Reshape((-1,)))
        model.add(Dense(1, activation="sigmoid"))
        # print("Starting Discriminator layers")
        # for layer in model.layers:
        #     print(layer.input_shape,layer.output_shape)
        # print("Ending Discriminator layers")
        return model

    def train(self, epochs, batch_size=128, gen_interval=50):
        # The test sets and classifier labels don't matter
        (X_train, _), (_, _) = fashion_mnist.load_data()

        # Rescaling images to [-1, 1] because of tanh
        X_train = X_train / 127.5 - 1.0

        # Adding a 4th dimension since keras expects 4D tensors
        X_train = np.expand_dims(X_train, axis=3)

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Generate a noise vector as input to the generator network
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            fake_imgs = self.generator.predict(noise)

            # Generate vectors of the correct labels to train the discriminator
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            #print(self.discriminator.metrics_names)
            #print(self.combined.metrics_names)

            real_loss = self.discriminator.train_on_batch(imgs, real)
            fake_loss = self.discriminator.train_on_batch(fake_imgs, fake)
            disc_loss = 0.5 * np.add(real_loss, fake_loss)
            #print(disc_loss)

            # Generate a new noise vector to train the generator on the combined model
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_loss = self.combined.train_on_batch(noise, real)
            #print(gen_loss)

            print("%d [Disc loss: %f, acc.: %.2f%%] [Gen loss: %f]" % (epoch, disc_loss[0], 100*disc_loss[1], gen_loss))

            if epoch % gen_interval == 0:
                self.gen_images(epoch)

    def gen_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_video(self):

        path = 'images/' # on Mac: right click on a folder, hold down option, and click "copy as pathname"

        image_folder = os.fsencode(path)

        filenames = []

        for file in os.listdir(image_folder):
            filename = os.fsdecode(file)
            if filename.endswith( ('.jpeg', '.png', '.gif') ):
                filenames.append(path+filename)

        filenames.sort() # this iteration technique has no built in order, so sort the frames

        images = list(map(lambda filename: imageio.imread(filename), filenames))

        imageio.mimsave(os.path.join('movie.gif'), images, duration = 0.04) # modify duration as needed



dcgan = GAN()
dcgan.train(10000)
dcgan.save_video()
