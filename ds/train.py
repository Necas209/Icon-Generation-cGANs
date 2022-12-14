import keras
import numpy as np

from ds.dataset import Icons50Dataset
from ds.gen import generate_real_samples, generate_fake_samples, generate_latent_points


# Needs to be adapted to the new dataset and the new model

def train(g_model: keras.Model, d_model: keras.Model, gan_model: keras.Model, dataset: Icons50Dataset, latent_dim: int,
          n_epochs: int, n_batch: int, n_classes: int) -> None:
    """ Train the GAN model """
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(dataset) / n_batch)
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, n_classes)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch, n_classes)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print(f'>{i + 1}, {j + 1}/{bat_per_epo}, d1={d_loss1:.3f}, d2={d_loss2:.3f} g={g_loss:.3f}')
    # save the generator model
    g_model.save('cgan_generator.h5')
