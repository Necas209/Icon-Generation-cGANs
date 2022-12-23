from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class Icons50Dataset:
    """ The Icons-50 dataset """
    images: np.ndarray[np.ndarray]
    """ image is a 3D numpy array of shape (32, 32, 3) """
    labels: np.ndarray[int]
    """ label is an integer in [0, 49] that represents the class of the image """
    classes: np.ndarray[str]
    """ classes is a list of strings that represent the classes of the dataset """
    subtypes: np.ndarray[str]
    """ subtype is a string that represents the icon subtype """
    styles: np.ndarray[str]
    """ style is a string that represents the icon style """
    renditions: np.ndarray[int]
    """ rendition is an integer in [0, 9] that represents the icon version """

    def __post_init__(self) -> None:
        self.__index = 0

    def preprocess(self) -> None:
        """ Preprocess the dataset """
        self.images = (self.images.astype(np.float32) - 127.5) / 127.5
        self.images = np.transpose(self.images, (0, 2, 3, 1))

    def shuffle(self) -> None:
        """ Shuffle the dataset """
        p = np.random.permutation(len(self))
        self.images = self.images[p]
        self.labels = self.labels[p]
        self.subtypes = self.subtypes[p]
        self.styles = self.styles[p]
        self.renditions = self.renditions[p]

    def summary(self, ordered: bool = False, first_k: int | None = None) -> None:
        """ Print a summary of the dataset """
        labels = np.unique(self.labels)
        details = [(label, np.count_nonzero(self.labels == label)) for label in labels]
        if ordered:
            details.sort(key=lambda x: x[1], reverse=True)
        print("Dataset summary:")
        print(f"Number of images: {len(self)}")
        print(f"Number of classes: {len(labels)}")
        print("Class distribution:")
        if first_k is not None:
            details = details[:first_k]
            print(f"First {first_k} classes represent {sum(x[1] for x in details) / len(self):.2%}% of the dataset")
        for label, count in details:
            print(f"Class {label} has {count} images")

    def filter(self, most_common: int | None = None) -> Icons50Dataset:
        """ Filter the dataset by the number of images per class """
        if most_common is None:
            return self
        labels, counts = np.unique(self.labels, return_counts=True)
        labels = labels[counts.argsort()[::-1]]
        labels = labels[:most_common]
        classes = self.classes[labels]
        # map labels to new labels
        label_map = {label: i for i, label in enumerate(labels)}
        labels = np.array([label_map.get(label, -1) for label in self.labels])
        return Icons50Dataset(
            images=self.images[labels != -1],
            labels=labels[labels != -1],
            classes=classes,
            subtypes=self.subtypes[labels != -1],
            styles=self.styles[labels != -1],
            renditions=self.renditions[labels != -1],
        )

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


def create_dataset(data_path: str | bytes | os.PathLike, classes_path: str | bytes | os.PathLike) -> Icons50Dataset:
    """ Create a dataset from a path """
    # Load the icons-50 dataset
    with open(data_path, 'rb') as f:
        icons = pickle.load(f)
    # Convert the lists to numpy arrays
    icons = {k: np.array(v) for k, v in icons.items()}
    # Read the classes
    classes = read_classes(classes_path)
    classes = np.array(classes)
    # Create the dataset
    return Icons50Dataset(
        images=icons["image"],
        labels=icons["class"],
        classes=classes,
        subtypes=icons["subtype"],
        styles=icons["style"],
        renditions=icons["rendition"]
    )


def read_classes(path: str | bytes | os.PathLike) -> list[str]:
    """ Read the classes from a file """
    with open(path, 'r') as f:
        return [line.strip() for line in f]


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
    print(f"Class: {dataset.classes[filter_label]} ({filter_label})")
    print("Subtypes:")
    for i, (_, subtype) in enumerate(types):
        print(f"{i:2d}. {subtype}")
