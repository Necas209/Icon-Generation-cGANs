import os

import hydra
from hydra.core.config_store import ConfigStore

from config.config import Icons50Config
from ds.dataset import Icons50Dataset, read_classes
from model.cgan import create_discriminator, create_generator, create_cgan
from model.train import save_models, train_cgan

cs = ConfigStore.instance()
cs.store(name="icons50_config", node=Icons50Config)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Icons50Config) -> None:
    """ Main function """
    # suppress TensorFlow INFO messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # Read classes
    classes = read_classes(cfg.paths.classes_path)
    # Load dataset
    dataset = Icons50Dataset.from_pickle(
        path=cfg.paths.data_path,
        classes=classes,
    )
    # Show dataset summary
    dataset.summary()
    # Preprocess the dataset
    dataset.preprocess()
    # Filter the dataset
    if cfg.params.top_k > 0:
        dataset = dataset.filter(cfg.params.top_k)
        cfg.params.num_classes = cfg.params.top_k
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
    history.plot()
    # save models
    save_models(
        generator=generator,
        discriminator=discriminator,
        cgan=cgan,
        path=cfg.paths.save_path if cfg.params.top_k == 0 else cfg.paths.filt_save_path,
    )


if __name__ == "__main__":
    main()
