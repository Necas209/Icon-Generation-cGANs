import numpy as np
from keras.models import Model

from ds.dataset import Icons50Dataset
from ds.generation import generate_real_samples, generate_fake_samples, generate_latent_points
from models.history import History


def summarize_performance(epoch: int, generator: Model, discriminator: Model, dataset: Icons50Dataset, latent_dim: int,
                          batch_size: int, num_classes: int) -> None:
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
          f'acc_real={acc_real:.2%}%, '
          f'acc_fake={acc_fake:.2%}%')


def train_cgan(cgan: Model, generator: Model, discriminator: Model, dataset: Icons50Dataset, latent_dim: int,
               epochs: int, batch_size: int, num_classes: int) -> History:
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
    for i in range(epochs):
        # enumerate batches over the training set
        for j in range(batches_per_epoch):
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
            d_loss = 0.5 * np.add(d_loss1, d_loss2)
            # calculate the discriminator accuracy
            d_acc = 0.5 * np.add(d_acc1, d_acc2)
            # summarize loss on this batch
            print(f'\r> Epoch {i + 1}: '
                  f'Batch {j + 1}/{batches_per_epoch}, '
                  f'disc_loss={d_loss:.3f}, '
                  f'disc_acc={d_acc:.3f}, '
                  f'gen_loss={g_loss:.3f}',
                  end='')
        print()
        # save metrics to history
        history.add(d_loss, d_acc, g_loss)
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, generator, discriminator, dataset, latent_dim, batch_size, num_classes)
    # save the generator model
    generator.save('cgan_generator.h5')

    return history
