import math

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Functional

from data.dataset import Icons50Dataset


def generate_real_samples(dataset: Icons50Dataset, num_samples: int):
    """ Select real samples from the dataset """
    # get images and labels
    images = dataset.images
    labels = dataset.labels
    # choose random instances
    ix = np.random.randint(0, images.shape[0], num_samples)
    # select images and labels
    x: np.ndarray = images[ix]
    labels: np.ndarray = labels[ix]
    # generate class labels
    y = np.ones((num_samples, 1))
    return (x, labels), y


def generate_latent_points(latent_dim: int, num_samples: int, num_classes: int):
    """ Generate points in the latent space as input for the generator """
    # generate points in the latent space
    x_input: np.ndarray = np.random.randn(latent_dim * num_samples)
    # reshape into a batch of inputs for the network
    z_input: np.ndarray = x_input.reshape(num_samples, latent_dim)
    # generate labels
    labels: np.ndarray = np.random.randint(0, num_classes, num_samples)
    return z_input, labels


def generate_fake_samples(generator: Functional, latent_dim: int, num_samples: int, num_classes: int):
    """ Generate fake samples using the generator """
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, num_samples, num_classes)
    # predict outputs
    images: np.ndarray = generator.predict_on_batch([z_input, labels_input])
    # create class labels
    y = np.zeros((num_samples, 1))
    return (images, labels_input), y


def generate_images(generator: Functional, label: int, latent_dim: int, num_images: int = 9) -> None:
    """ Create a plot of generated images given a label """
    # generate points in latent space
    z_input, _ = generate_latent_points(latent_dim, num_images, 1)
    # specify labels
    labels = np.asarray([label for _ in range(num_images)])
    # generate images
    x = generator.predict([z_input, labels])
    # scale from [-1, 1] to [0, 255]
    x = (x + 1) / 2.0
    # determine the plot size
    n_cols = int(math.sqrt(num_images))
    n_rows = int(math.ceil(num_images / n_cols))
    # plot images
    for i in range(num_images):
        # define subplot
        plt.subplot(n_rows, n_cols, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(x[i])
    plt.show()
