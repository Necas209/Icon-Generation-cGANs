import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from matplotlib import pyplot as plt
from tensorflow import keras

from dataset import get_dataset_from_dict


def main() -> None:
    # Load the dataset
    icons = np.load("Icons-50.npy", allow_pickle=True).item()

    # Create a dataset from the icons
    dataset = get_dataset_from_dict(icons)

    # Create TF dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices((dataset.images, dataset.labels))

    # Load the model from the Hugging Face Hub
    model: keras.Model = from_pretrained_keras("keras-io/conditional-gan")
    model.summary()

    # Generate a random noise vector
    noise = np.random.normal(0, 1, (1, 100))

    # Generate a random class label
    label = np.random.randint(0, 50, 1)

    # Generate an image
    generated_image = model.predict([noise, label])

    # Plot the image
    plt.imshow(generated_image[0, :, :, 0], cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
