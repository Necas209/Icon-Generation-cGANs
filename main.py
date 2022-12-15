from __future__ import annotations

import hydra
from hydra.core.config_store import ConfigStore
from keras.models import load_model

from config import Icons50Config
from ds.dataset import create_dataset, print_labels
from ds.generation import generate_images
from models.discriminator import create_discriminator
from models.gan import create_gan
from models.generator import create_generator
from models.train import train_gan

cs = ConfigStore.instance()
cs.store(name="mnist_config", node=Icons50Config)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Icons50Config) -> None:
    # Create the dataset
    dataset = create_dataset(cfg.file_path)

    # Shuffle the dataset
    if cfg.params.shuffle:
        dataset.shuffle()

    # create the discriminator
    d_model = create_discriminator(
        image_size=cfg.params.image_size,
        channels=cfg.params.channels,
        n_classes=cfg.params.n_classes,
        lr=cfg.optimizer.lr,
        beta_1=cfg.optimizer.beta_1,
    )

    # create the generator
    g_model = create_generator(
        latent_dim=cfg.params.latent_dim,
        n_classes=cfg.params.n_classes,
    )

    # create the gan
    gan_model = create_gan(
        g_model, d_model,
        lr=cfg.optimizer.lr,
        beta_1=cfg.optimizer.beta_1
    )
    gan_model.summary()

    # train model
    train_gan(
        g_model, d_model, gan_model,
        dataset=dataset,
        latent_dim=cfg.params.latent_dim,
        n_epochs=cfg.params.epochs,
        n_batch=cfg.params.batch_size,
        n_classes=cfg.params.n_classes
    )

    # load saved generator model
    g_model = load_model('cgan_generator.h5')

    # Read the label from user input
    label = int(input("Enter the label: "))
    print_labels(dataset, label)

    # generate images
    generate_images(
        g_model,
        latent_dim=cfg.params.latent_dim,
        n_images=9,
        label=label
    )


if __name__ == "__main__":
    main()
