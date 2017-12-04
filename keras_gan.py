
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from matplotlib.image import imread

import sys,os
import numpy as np
from PIL import Image
from random import shuffle


#DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = "/Users/keganrabil/Desktop/WIP/tensorflow"
DATA_PATH = os.path.join(DIR_PATH,"data","celeb")


# Prepare data
class CelebImageBatcher:
    def __init__(self,data_path,batch_size=128,scale=0.25):
        self.scale = scale
        self.start = 0
        self.end = -1
        self.batch_size = batch_size
        self.path = data_path
        self.data_dict = {}
        count = 0
        for root, dirs, files in os.walk(self.path):
            shuffle(files)
            for name in files:
                self.data_dict[count] = os.path.join(root, name)
                count += 1
        self.end = count

    def next(self):
        batch = []
        for i in range(self.start,self.start+self.batch_size):
            #image = imread(self.data_dict[i])
            img = Image.open(self.data_dict[i])
            w_scale = 48#int(img.size[0] * self.scale)
            h_scale = 64#int(img.size[1] * self.scale)
            img = img.resize((w_scale,h_scale), Image.ANTIALIAS)
            batch.append(np.asarray(img))
        self.start = (self.start + self.batch_size) % self.end
        #return np.asarray(batch)
        return (np.asarray(batch).astype(np.float32) - 127.5) / 127.5

    def test(self):
        r,c = 5,5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        b = self.next()
        print(b.shape)
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(b[cnt, :,:,:])#, cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        plt.show()


class GAN():
    def __init__(self):
        self.img_rows = 64#28
        self.img_cols = 48#28
        self.channels = 3#1
        self.reduce_rows = 4
        self.reduce_cols = 3
        self.reduce_channels = 48

        self.kernel_size = [5,5]
        self.strides = (2,2)
        self.padding = "same"#"valid"

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.reduce_shape = (self.reduce_rows, self.reduce_cols, self.reduce_channels)
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.reduce_channels*self.reduce_rows*self.reduce_cols,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (self.reduce_channels*self.reduce_rows*self.reduce_cols,)

        model = Sequential()

        #model.add(Dense(256, input_shape=noise_shape))
        model.add(Reshape(input_shape=noise_shape,target_shape=self.reduce_shape))
        model.add(Conv2DTranspose(24, self.kernel_size, strides=self.strides, padding=self.padding, data_format="channels_last"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(momentum=0.9))

        #model.add(Dense(512))
        model.add(Conv2DTranspose(12, self.kernel_size, strides=self.strides, padding=self.padding, data_format="channels_last"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(momentum=0.9))

        #model.add(Dense(1024))
        model.add(Conv2DTranspose(6, self.kernel_size, strides=self.strides, padding=self.padding, data_format="channels_last"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(momentum=0.9))

        #model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Conv2DTranspose(3, self.kernel_size, strides=self.strides, padding=self.padding, activation='tanh', data_format="channels_last"))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        #model.add(Flatten(input_shape=img_shape))
        #model.add(Dense(512))
        model.add(Conv2D(input_shape=img_shape,filters=6, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, data_format="channels_last"))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(12, self.kernel_size, strides=self.strides, padding=self.padding, data_format="channels_last"))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(24, self.kernel_size, strides=self.strides, padding=self.padding, data_format="channels_last"))
        model.add(LeakyReLU(alpha=0.1))

        #model.add(Conv2D(48, self.kernel_size, strides=self.strides, padding=self.padding, data_format="channels_last"))
        #model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())#input_shape=self.reduce_shape))

        #model.add(Dense(256))
        #model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)
        batcher = CelebImageBatcher(DATA_PATH,half_batch)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            #idx = np.random.randint(0, X_train.shape[0], half_batch)
            #imgs = X_train[idx]
            imgs = batcher.next()

            noise = np.random.normal(0, 1, (half_batch, self.reduce_channels*self.reduce_rows*self.reduce_cols))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.reduce_channels*self.reduce_rows*self.reduce_cols))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.reduce_channels*self.reduce_rows*self.reduce_cols))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])#, cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan/images/face_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, save_interval=100)
