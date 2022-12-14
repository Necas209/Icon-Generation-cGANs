from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Icons50Dataset:
    """ The Icons-50 dataset """
    images: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.__index = 0
        self.images = self.images.astype('float32')
        self.images = (self.images - 127.5) / 127.5
        self.images = np.transpose(self.images, (0, 2, 3, 1))

    def shuffle(self) -> None:
        """ Shuffle the dataset """
        p = np.random.permutation(len(self))
        self.images = self.images[p]
        self.labels = self.labels[p]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.images[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.__index >= len(self):
            self.__index = 0
            raise StopIteration
        else:
            self.__index += 1
            return self[self.__index - 1]


def create_dataset(path: str) -> Icons50Dataset:
    """ Create a ds from a path """
    # Load the numpy arrays
    icons: dict | Any = np.load(path, allow_pickle=True).item()
    # Get the images and labels
    images: np.ndarray = icons["image"]
    labels = np.array(icons["class"])
    # Create the ds
    dataset = Icons50Dataset(images, labels)
    return dataset
