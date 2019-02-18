from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv2DTranspose, Activation, Reshape, Input
from keras.layers import BatchNormalization, Dropout
from keras.layers import LeakyReLU
from keras.datasets import mnist
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

        # z = Input(shape=(self.latent_dim,))
        # img = self.generator(z)





        self.combined = Sequential()
        self.combined.add(self.generator)
        self.discriminator.trainable = False

        # validity = self.discriminator(img)
        # self.combined = Model(z, validity)

        self.combined.add(self.discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optim)






    def build_generator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))
        model.add(Dense(768))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))
        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))
        # model.add(Dense(2048))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.input_shape), activation='tanh'))
        model.add(Reshape(self.input_shape))
        # model.add(Dense(1024, input_dim=self.latent_dim))
        # model.add(Reshape((4, 4, -1)))
        # model.add(Conv2DTranspose(128, 4, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(128, 4, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(128, 4, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(128, 4, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(128, 4, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(128, 4, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(128, 4, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(64, 5, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(64, 5, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(64, 5, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(64, 5, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(64, 5, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2DTranspose(self.num_channels, 4, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Activation('tanh'))
        print("Starting Generator layers")
        for layer in model.layers:
            print(layer.input_shape, layer.output_shape)
        print("Ending Generator layers")
        return model
        # noise = Input(shape=(self.latent_dim,))
        # img = model(noise)
        # return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        # model.add(Dense(2048, input_shape=self.input_shape))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128,input_shape=self.input_shape ))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))
        model.add(Dense(16))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(128))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(128, 4,input_shape=self.input_shape, data_format="channels_last"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(128, 7, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(128, 7, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(128, 7, data_format="channels_last"))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(64, 4, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(64, 4, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(64, 4, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(64, 4, data_format="channels_last", padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Conv2D(64, 4, data_format="channels_last", padding='same'))
        model.add(Reshape((-1,)))
        model.add(Dense(1, activation="sigmoid"))
        print("Starting Discriminator layers")
        for layer in model.layers:
            print(layer.input_shape,layer.output_shape)
        print("Ending Discriminator layers")
        return model
        # img = Input(shape=self.input_shape)
        # validity = model(img)
        #
        # return Model(img, validity)

    def train(self, epochs, batch_size=128, gen_interval=50):
        # The test sets and classifier labels don't matter
        (X_train, Y_train), (_, _) = mnist.load_data()

        # Rescaling images to [-1, 1] because of tanh
        X_train = X_train / 127.5 - 1.0

        # Training on only images of 3
        X_train = X_train[Y_train==3]

        # Adding a 4th dimension since keras expects 4D tensors
        X_train = np.expand_dims(X_train, axis=3)

        prev_disc_acc = 0
        prev_gen_loss = 100
        count = 0

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            #idx[0] =  1
            imgs = X_train[idx]

            # Adding decaying noise to the real discriminator inputs and then
            # renormalizing to the [-1, 1] range
            max_noise = 0.5 ** (epoch / 10.0)
            imgs += np.random.random(imgs.shape) * 2 * max_noise - max_noise
            imgs = 2 * (imgs - np.min(imgs))/np.ptp(imgs)-1

            # Generate a noise vector as input to the generator network
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            fake_imgs = self.generator.predict(noise)

            # Adding decaying noise to the fake discriminator inputs and then
            # renormalizing to the [-1, 1] range
            fake_imgs += np.random.random(fake_imgs.shape) * 2 * max_noise - max_noise
            fake_imgs = 2 * (fake_imgs - np.min(fake_imgs))/np.ptp(fake_imgs)-1

            # Generate vectors of the correct labels to train the discriminator
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # if epoch == 0:
            #     print("Real:",real)
            #     print("Fake:",fake)

            #print(self.discriminator.metrics_names)
            #print(self.combined.metrics_names)

            if prev_disc_acc < 0.6 or count > 50:
                real_loss = self.discriminator.train_on_batch(imgs, real)
                fake_loss = self.discriminator.train_on_batch(fake_imgs, fake)
                disc_loss = 0.5 * np.add(real_loss, fake_loss)
                print(disc_loss)
                prev_disc_acc = disc_loss[1]
                count = 0
            else:
                real_loss = self.discriminator.evaluate(imgs, real)
                fake_loss = self.discriminator.evaluate(fake_imgs, fake)
                disc_loss = 0.5 * np.add(real_loss, fake_loss)
                prev_disc_acc = disc_loss[1]

            # Generate a new noise vector to train the generator on the combined model
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_loss = self.combined.train_on_batch(noise, real)
            #print(gen_loss)

            print("%d [Disc loss: %f, acc.: %.2f%%] [Gen loss: %f]" % (epoch, disc_loss[0], 100*disc_loss[1], gen_loss))

            if epoch % gen_interval == 0:
                self.gen_images(epoch)

            count += 1
            prev_gen_loss = gen_loss

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
        fig.savefig("images11/%d.png" % epoch)
        plt.close()

    def save_video(self):

        path = 'images11/' # on Mac: right click on a folder, hold down option, and click "copy as pathname"

        image_folder = os.fsencode(path)

        filenames = []

        for file in os.listdir(image_folder):
            filename = os.fsdecode(file)
            if filename.endswith( ('.jpeg', '.png', '.gif') ):
                filenames.append(path+filename)

        filenames.sort() # this iteration technique has no built in order, so sort the frames

        images = list(map(lambda filename: imageio.imread(filename), filenames))

        imageio.mimsave(os.path.join('movie11.gif'), images, duration = 0.04) # modify duration as needed



dcgan = GAN()
dcgan.train(10000)
dcgan.save_video()
