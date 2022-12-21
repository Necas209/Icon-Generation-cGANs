from keras import Input
from keras.layers import Concatenate, Dense, Conv2D, LeakyReLU, Embedding, Reshape, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_generator(latent_dim: int, num_classes: int) -> Model:
    """ Create a generator model """
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(num_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 8 * 8
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((8, 8, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 8x8 image
    n_nodes = 128 * 8 * 8
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((8, 8, 128))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 16x16
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 32x32
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(3, (8, 8), activation='tanh', padding='same')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer, name='generator')
    return model


# define the combined generator and discriminator model, for updating the generator
def create_discriminator(image_size: int, channels: int, num_classes: int, lr: float, beta_1: float) -> Model:
    """ Create a discriminator model """
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(num_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = image_size * image_size
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((image_size, image_size, 1))(li)
    # image input
    in_image = Input(shape=(image_size, image_size, channels))
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    # downsample to 16x16
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample to 8x8
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout layer
    fe = Dropout(0.4)(fe)
    # output layer nodes
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_image, in_label], out_layer, name='discriminator')
    # compile model
    opt = Adam(learning_rate=lr, beta_1=beta_1)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    return model


def create_cgan(generator: Model, discriminator: Model, lr: float, beta_1: float) -> Model:
    """ Create a cGAN model """
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = generator.input
    # get image output from the generator model
    gen_output = generator.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = discriminator([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output, name='cgan')
    # compile model
    opt = Adam(learning_rate=lr, beta_1=beta_1)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt
    )
    return model
