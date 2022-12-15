import math

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model

from ds.dataset import Icons50Dataset


def generate_real_samples(dataset: Icons50Dataset, n_samples: int) -> tuple[list[np.ndarray], np.ndarray]:
    """ Select real samples from the ds """
    # get images and labels
    images = dataset.images
    labels = dataset.labels
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    x, labels = images[ix], labels[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return [x, labels], y


def generate_latent_points(latent_dim: int, n_samples: int, n_classes: int) -> list[np.ndarray]:
    """ Generate points in the latent space as input for the generator """
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]


def generate_fake_samples(generator: Model,
                          latent_dim: int, n_samples: int, n_classes: int) -> tuple[list[np.ndarray], np.ndarray]:
    """ Generate fake samples using the generator """
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
    # predict outputs
    images = generator.predict_on_batch([z_input, labels_input])
    # create class labels
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y


def generate_images(g_model: Model, label: int, latent_dim: int, n_images: int = 9) -> None:
    """ Generate a plot of generated images """
    # generate points in latent space
    z_input, _ = generate_latent_points(latent_dim, n_images, 1)
    # specify labels
    labels = np.asarray([label for _ in range(n_images)])
    # generate images
    x = g_model.predict([z_input, labels])
    # scale from [-1, 1] to [0, 255]
    x = (x + 1) / 2.0
    # determine the plot size
    n_cols = int(math.sqrt(n_images))
    n_rows = int(math.ceil(n_images / n_cols))
    # plot images
    for i in range(n_images):
        # define subplot
        plt.subplot(n_rows, n_cols, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(x[i])
    plt.show()
