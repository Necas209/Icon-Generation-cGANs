import os

import hydra
from hydra.core.config_store import ConfigStore

from config.config import Icons50Config
from ds.dataset import Icons50Dataset, read_classes, read_label
from ds.generation import generate_images
from model.train import load_generator

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
    # Filter the dataset
    if cfg.params.top_k > 0:
        dataset = dataset.filter(cfg.params.top_k)
        cfg.params.num_classes = cfg.params.top_k
    # load saved generator model
    generator = load_generator(path=cfg.paths.filt_save_path)
    # Read the label from user input
    label = read_label(num_classes=cfg.params.num_classes)
    dataset.print_subtypes(label)
    # generate images
    generate_images(
        generator=generator,
        label=label,
        latent_dim=cfg.params.latent_dim
    )


if __name__ == "__main__":
    main()
