from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Icons50Dataset:
    """ The Icons-50 dataset """
    images: np.ndarray[np.ndarray]
    """ image is a 3D numpy array of shape (32, 32, 3) """
    labels: np.ndarray[int]
    """ label is an integer in [0, 49] that represents the class of the image """
    subtypes: np.ndarray[str]
    """ subtype is a string that represents the icon subtype """
    styles: np.ndarray[str]
    """ style is a string that represents the icon style """
    renditions: np.ndarray[int]
    """ rendition is an integer in [0, 9] that represents the icon version """

    def __post_init__(self) -> None:
        self.__index = 0
        self.images = self.images.astype(np.float32)
        self.images = (self.images - 127.5) / 127.5
        self.images = np.transpose(self.images, (0, 2, 3, 1))

    def shuffle(self) -> None:
        """ Shuffle the dataset """
        p = np.random.permutation(len(self))
        self.images = self.images[p]
        self.labels = self.labels[p]
        self.subtypes = self.subtypes[p]
        self.styles = self.styles[p]
        self.renditions = self.renditions[p]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.images[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Icons50Dataset:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.__index >= len(self):
            self.__index = 0
            raise StopIteration
        else:
            self.__index += 1
            return self[self.__index - 1]


def create_dataset(path: str | bytes | os.PathLike) -> Icons50Dataset:
    """ Create a dataset from a path """
    # Load the icons-50 dataset
    icons: dict | Any
    # check if python version is 3.8 or lower
    if sys.version_info < (3, 9):
        icons = np.load(path).item()
    else:
        icons = np.load(path, allow_pickle=True).item()
    # Convert the lists to numpy arrays
    icons = {k: np.array(v) for k, v in icons.items()}
    # Create the dataset
    return Icons50Dataset(
        images=icons["image"],
        labels=icons["class"],
        subtypes=icons["subtype"],
        styles=icons["style"],
        renditions=icons["rendition"]
    )


def print_labels(dataset: Icons50Dataset, filter_label: int | None = None) -> None:
    """ Print the labels and subtypes """
    # Zip the labels and subtypes
    types = list(zip(dataset.labels, dataset.subtypes))
    # Filter out the duplicates
    types = list(set(types))
    # Sort the types
    types.sort(key=lambda x: (x[0], x[1]))
    # If a label is specified, print only that label
    if filter_label is not None:
        types = [x for x in types if x[0] == filter_label]
    for label, subtype in types:
        print(f"{label}: {subtype}")
