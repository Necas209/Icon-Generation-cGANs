from __future__ import annotations

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import Icons50Config
from ds.dataset import create_dataset
from model.discriminator import create_discriminator
from model.gan import create_gan
from model.train import train_gan
from model.generator import create_generator

cs = ConfigStore.instance()
cs.store(name="mnist_config", node=Icons50Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Icons50Config) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Create the ds
    dataset = create_dataset(cfg.file_path)

    # Shuffle the ds
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


if __name__ == "__main__":
    main()
