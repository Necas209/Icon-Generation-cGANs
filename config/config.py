from dataclasses import dataclass


@dataclass
class Optimizer:
    lr: float
    beta_1: float


@dataclass
class Params:
    latent_dim: int
    num_classes: int
    image_size: int
    channels: int
    epochs: int
    batch_size: int
    shuffle: bool


@dataclass
class Paths:
    data_path: str
    save_path: str
    filt_save_path: str


@dataclass
class Icons50Config:
    paths: Paths
    params: Params
    optim: Optimizer
