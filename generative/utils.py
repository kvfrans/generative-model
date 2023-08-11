import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_image(image):
    """Plot a single MNIST image."""
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def plot_image_row(images):
    """Plot a row of MNIST images."""
    images = jnp.clip(images, 0, 1)
    plt.figure(figsize=(4*len(images), 4))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.show()

def plot_image_grid(images):
    """Plot a grid of MNIST images."""
    images = jnp.clip(images, 0, 1)
    plt.figure(figsize=(4*images.shape[0], 4*images.shape[1]))
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            plt.subplot(images.shape[0], images.shape[1], i * images.shape[1] + j + 1)
            plt.imshow(images[i, j])
            plt.axis("off")
    plt.show()

def plot_frequency_table(frequency_table):
    # plot frequency table as a 2D histogram. on the left axis is ["0A", "0B", "1", "2"]. on the bottom axis is ["Red", "RedBlue", "BlueRed", "Blue", "Green"].
    plt.figure(figsize=(8, 8))
    plt.imshow(frequency_table, interpolation='nearest', aspect=0.5)
    # label it.

    # Label each cell with the number of times that character/color combination appeared in the dataset.
    for i in range(frequency_table.shape[0]):
        for j in range(frequency_table.shape[1]):
            plt.text(j, i, frequency_table[i, j], ha="center", va="center", color="black", fontsize=20)

    plt.ylabel("Color", fontsize=20)
    plt.xlabel("Character", fontsize=20)
    plt.yticks([0, 1, 2, 3, 4], ["Red", "RedBlue", "BlueRed", "Blue", "Green"])
    plt.xticks([0, 1, 2, 3, 4, 5], ["0A", "0B", "1", "2", "3", "4"])
    plt.title("Frequency per character/color combination", fontsize=20)


    # fontsize 20 for ticks
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Set minimum and maximum values for the colorbar.
    plt.clim(0, None)

    # show with figsize (10, 10)
    plt.show()

def plot_error_table(error_table, frequency_table):
    # plot frequency table as a 2D histogram. on the left axis is ["0A", "0B", "1", "2"]. on the bottom axis is ["Red", "RedBlue", "BlueRed", "Blue", "Green"].
    errors_scaled = error_table / (frequency_table + 1e-5)
    plt.figure(figsize=(8, 8))
    plt.imshow(errors_scaled, interpolation='nearest', aspect=0.5)
    # label it.

    # Label each cell with the number of times that character/color combination appeared in the dataset.
    for i in range(errors_scaled.shape[0]):
        for j in range(errors_scaled.shape[1]):
            plt.text(j, i, "{:.2f}".format(errors_scaled[i, j]), ha="center", va="center", color="black", fontsize=20)

    plt.ylabel("Color", fontsize=20)
    plt.xlabel("Character", fontsize=20)
    plt.yticks([0, 1, 2, 3, 4], ["Red", "RedBlue", "BlueRed", "Blue", "Green"])
    plt.xticks([0, 1, 2, 3, 4, 5], ["0A", "0B", "1", "2", "3", "4"])
    plt.title("Error rate per character/color combination", fontsize=20)

    # fontsize 20 for ticks
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Set minimum and maximum values for the colorbar.
    plt.clim(0, None)

    # show with figsize (10, 10)
    plt.show()

def plot_examples_table(examples_table):
    examples_flat = jnp.concatenate([examples_table[i] for i in range(5)], axis=1)
    examples_flat = jnp.concatenate([examples_flat[i] for i in range(6)], axis=1)
    plt.figure(figsize=(8, 8))
    plt.imshow(examples_flat)
    plt.axis("off")
    plt.show()