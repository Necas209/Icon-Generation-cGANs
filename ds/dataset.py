from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Icons50Dataset:
    """ The Icons-50 dataset
    Attributes:
        images: NumPy array of shape (no_images, 32, 32, 3) containing the images
        labels: NumPy array of shape (no_images,) containing the class label for each image
        classes: NumPy array of shape (no_classes,) containing the class names
        subtypes: NumPy array of shape (no_images,) containing the subtype of each image
        styles: NumPy array of shape (no_images,) containing the style of each image
        renditions: NumPy array of shape (no_images,) containing the version of each image
    """
    images: np.ndarray[np.ndarray]
    labels: np.ndarray[int]
    classes: np.ndarray[str]
    subtypes: np.ndarray[str]
    styles: np.ndarray[str]
    renditions: np.ndarray[int]

    def __len__(self) -> int:
        """ Return the number of images in the dataset """
        return len(self.images)

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

    def summary(self, top_k: int | None = None) -> None:
        """ Print a summary of the dataset """
        no_images = len(self)
        no_classes = len(self.classes)
        print("Dataset summary:")
        print(f"Number of images: {no_images}")
        print(f"Number of classes: {no_classes}")
        print("Class distribution:")
        if top_k is not None:
            _, counts = np.unique(self.labels, return_counts=True)
            counts = counts[counts.argsort()[::-1]]
            counts = counts[:top_k]
            print(f"Top {top_k} classes account for {sum(counts)} images ({sum(counts) / no_images:%}%)")
        plt.hist(self.labels, bins=no_classes)
        plt.xlabel("Class label")
        plt.ylabel("Number of images")
        plt.show()

    def filter(self, top_k: int) -> Icons50Dataset:
        """ Filter the dataset to contain only the top k classes """
        # Get the top k classes
        labels, counts = np.unique(self.labels, return_counts=True)
        # Sort the classes by count
        labels = labels[counts.argsort()[::-1]]
        # Get the top k labels
        labels = labels[:top_k]
        # Filter the dataset
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

    def print_classes(self) -> None:
        """ Print the classes of the dataset """
        for label in range(len(self.classes)):
            print(f"{label}: {self.classes[label]}")

    def print_subtypes(self, label: int | None = None) -> None:
        """ Print the subtypes of the dataset for a given class """
        if label is None:
            for label in range(len(self.classes)):
                print(f"Class {label}: {self.classes[label]}")
                self.print_subtypes(label)
            return
        subtypes, counts = np.unique(self.subtypes[self.labels == label], return_counts=True)
        print(f"Subtypes for class {label}: {self.classes[label]}")
        for subtype, count in zip(subtypes, counts):
            print(f"{subtype}: {count}")

    @staticmethod
    def from_pickle(path: str | bytes | os.PathLike, classes: list[str]) -> Icons50Dataset:
        """ Create a dataset from a path """
        # Load the icons-50 dataset
        with open(path, 'rb') as f:
            icons = pickle.load(f)
        # Convert the lists to numpy arrays
        icons = {k: np.array(v) for k, v in icons.items()}
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
