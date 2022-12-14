import numpy as np
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from ds.dataset import Icons50Dataset
from ds.generation import generate_real_samples, generate_fake_samples, generate_latent_points


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model: Model, d_model: Model, lr: float, beta_1: float) -> Model:
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(lr=lr, beta_1=beta_1)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt
    )
    return model


def summarize_performance(i: int, g_model: Model, d_model: Model, dataset: Icons50Dataset, latent_dim: int,
                          n_batch: int, n_classes: int) -> None:
    """ Summarize model performance """
    # prepare real samples
    [X_real, labels_real], y_real = generate_real_samples(dataset, n_batch)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate([X_real, labels_real], y_real, verbose=0)
    # prepare fake examples
    [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, n_batch, n_classes)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate([X_fake, labels], y_fake, verbose=0)
    # summarize discriminator performance
    print(f'>Accuracy real: {acc_real:.2%}%, fake: {acc_fake:.2%}%')
    # save the generator model tile file
    # noinspection PyUnusedLocal
    filename = f'cgan_generator_model_{i + 1}.h5'
    # g_model.save(filename)


def train_gan(g_model: Model, d_model: Model, gan_model: Model, dataset: Icons50Dataset, latent_dim: int, n_epochs: int,
              n_batch: int, n_classes: int) -> None:
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
            print(f'>{i + 1}, {j + 1}/{bat_per_epo}, d1={d_loss1:.3f}, d2={d_loss2:.3f} g={g_loss:.3f}', end='\r')
        print()
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim, n_batch, n_classes)
    # save the generator model
    g_model.save('cgan_generator.h5')
