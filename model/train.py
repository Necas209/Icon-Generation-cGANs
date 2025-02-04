from __future__ import annotations

import os

import numpy as np
from keras.models import Functional, load_model

from ds.dataset import Icons50Dataset
from ds.generation import generate_real_samples, generate_fake_samples, generate_latent_points
from model.history import History


def summarize_performance(epoch: int, generator: Functional, discriminator: Functional, dataset: Icons50Dataset,
                          latent_dim: int, batch_size: int, num_classes: int) -> None:
    """ Summarize model performance """
    # prepare real samples
    (X_real, labels_real), y_real = generate_real_samples(dataset, batch_size)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate([X_real, labels_real], y_real, verbose=0)
    # prepare fake examples
    (X_fake, labels), y_fake = generate_fake_samples(generator, latent_dim, batch_size, num_classes)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate([X_fake, labels], y_fake, verbose=0)
    # summarize discriminator performance
    print(f'> Epoch {epoch + 1}: '
          f'acc_real={acc_real:.2%}, '
          f'acc_fake={acc_fake:.2%}')


def train_cgan(cgan: Functional, generator: Functional, discriminator: Functional, dataset: Icons50Dataset,
               latent_dim: int, epochs: int, batch_size: int, num_classes: int) -> History:
    """ Train the cGAN model """
    # create history of performance for plotting
    history = History()
    # define metrics
    d_loss = 0.0
    d_acc = 0.0
    g_loss = 0.0
    # calculate the number of batches per training epoch
    batches_per_epoch = int(len(dataset) / batch_size)
    # calculate the size of half a batch of samples
    half_batch = int(batch_size / 2)
    # manually enumerate epochs
    for epoch in range(epochs):
        # enumerate batches over the training set
        for batch in range(batches_per_epoch):
            # get randomly selected 'real' samples
            (X_real, labels_real), y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, d_acc1 = discriminator.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            (X_fake, labels), y_fake = generate_fake_samples(generator, latent_dim, half_batch, num_classes)
            # update discriminator model weights
            d_loss2, d_acc2 = discriminator.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            z_input, labels_input = generate_latent_points(latent_dim, batch_size, num_classes)
            # create inverted labels for the fake samples
            y_gan = np.ones((batch_size, 1))
            # update the generator via the discriminator's error
            g_loss = cgan.train_on_batch([z_input, labels_input], y_gan)
            # calculate the discriminator loss
            d_loss = (d_loss1 + d_loss2) / 2.0
            # calculate the discriminator accuracy
            d_acc = (d_acc1 + d_acc2) / 2.0
            # summarize loss on this batch
            print(f'\r> Epoch {epoch + 1}: '
                  f'Batch {batch + 1}/{batches_per_epoch}, '
                  f'disc_loss={d_loss:.3f}, '
                  f'disc_acc={d_acc:.3f}, '
                  f'gen_loss={g_loss:.3f}',
                  end='')
        print()
        # save metrics to history
        history.add(d_loss, d_acc, g_loss)
        # evaluate the model performance, sometimes
        if (epoch + 1) % 10 == 0:
            summarize_performance(epoch, generator, discriminator, dataset, latent_dim, batch_size, num_classes)
    return history


def save_models(generator: Functional, discriminator: Functional, cgan: Functional,
                path: str | bytes | os.PathLike) -> None:
    """ Save the models to file in SavedModel format """
    generator.save(os.path.join(path, 'generator'))
    discriminator.save(os.path.join(path, 'discriminator'))
    cgan.save(os.path.join(path, 'cgan'))


def load_generator(path: str | bytes | os.PathLike) -> Functional:
    """ Load the generator model from file """
    path = os.path.join(path, 'generator')
    return load_model(path)
