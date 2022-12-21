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
class Icons50Config:
    file_path: str
    save_path: str
    params: Params
    optim: Optimizer
