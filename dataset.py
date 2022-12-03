from dataclasses import dataclass

import numpy as np

SampleType = tuple[np.ndarray[np.ndarray], int, str, str, int]


@dataclass
class Dataset:
    images: np.ndarray[np.ndarray]  # image is a 3x32x32 array
    labels: np.ndarray[int]  # class labels
    subtypes: np.ndarray[str]  # subtype is a string that indicates the icon's subtype
    styles: np.ndarray[str]  # style is a string that indicates the icon's style
    renditions: np.ndarray[int]  # rendition is a string that indicates the icon's version

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> SampleType:
        return (
            self.images[index],
            self.labels[index],
            self.subtypes[index],
            self.styles[index],
            self.renditions[index]
        )

    def __iter__(self) -> SampleType:
        for index in range(len(self)):
            yield self[index]


def get_dataset_from_dict(icons: dict) -> Dataset:
    icons = {k: np.array(v) for k, v in icons.items()}
    return Dataset(
        images=icons["image"],
        labels=icons["class"],
        subtypes=icons["subtype"],
        styles=icons["style"],
        renditions=icons["rendition"]
    )
