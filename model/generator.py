from keras import Input
from keras.layers import Concatenate, Dense, Conv2D, LeakyReLU, Embedding, Reshape, Conv2DTranspose
from keras.models import Model


def define_generator(latent_dim: int, n_classes: int = 10) -> Model:
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
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
    model = Model([in_lat, in_label], out_layer)
    return model
