import os

import hydra
from hydra.core.config_store import ConfigStore
from keras.models import load_model

from config.config import Icons50Config
from data.dataset import create_dataset, print_labels
from data.generation import generate_images
from model.cgan import create_cgan, create_generator, create_discriminator
from model.history import plot_history
from model.train import train_cgan, save_models

cs = ConfigStore.instance()
cs.store(name="icons50_config", node=Icons50Config)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Icons50Config) -> None:
    """ Main function """
    # suppress TensorFlow INFO messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # Create the dataset
    dataset = create_dataset(cfg.file_path)
    # Shuffle the dataset
    if cfg.params.shuffle:
        dataset.shuffle()
    # create the discriminator
    discriminator = create_discriminator(
        image_size=cfg.params.image_size,
        channels=cfg.params.channels,
        num_classes=cfg.params.num_classes,
        lr=cfg.optim.lr,
        beta_1=cfg.optim.beta_1,
    )
    discriminator.summary()
    # create the generator
    generator = create_generator(
        latent_dim=cfg.params.latent_dim,
        num_classes=cfg.params.num_classes,
    )
    generator.summary()
    # create the gan
    cgan = create_cgan(
        generator=generator,
        discriminator=discriminator,
        lr=cfg.optim.lr,
        beta_1=cfg.optim.beta_1
    )
    cgan.summary()
    # train model
    history = train_cgan(
        cgan=cgan,
        generator=generator,
        discriminator=discriminator,
        dataset=dataset,
        latent_dim=cfg.params.latent_dim,
        epochs=cfg.params.epochs,
        batch_size=cfg.params.batch_size,
        num_classes=cfg.params.num_classes
    )
    # plot history
    plot_history(history)
    # save models
    save_models(generator, discriminator, cgan, cfg.save_path)
    # load saved generator model
    generator = load_model('cgan_generator.h5')
    # Read the label from user input
    label = int(input("Enter the label: "))
    print_labels(dataset, label)
    # generate images
    generate_images(
        generator=generator,
        label=label,
        latent_dim=cfg.params.latent_dim
    )


if __name__ == "__main__":
    main()
