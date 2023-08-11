import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import gzip
import struct
import array
from os import path
import os
import urllib.request
import jax
import jax.numpy as jnp

_DATA = "data/"

def _download(url, filename):
	if not path.exists(_DATA):
		os.makedirs(_DATA)
	out_file = path.join(_DATA, filename)
	if not path.isfile(out_file):
		urllib.request.urlretrieve(url, out_file)
		print(f"downloaded {url} to {_DATA}")

def mnist_raw():
	"""Download and parse the raw MNIST dataset."""
	# CVDF mirror of http://yann.lecun.com/exdb/mnist/
	base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

	def parse_labels(filename):
		with gzip.open(filename, "rb") as fh:
			_ = struct.unpack(">II", fh.read(8))
			return jnp.array(array.array("B", fh.read()), dtype=np.uint8)

	def parse_images(filename):
		with gzip.open(filename, "rb") as fh:
			_, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
			return jnp.array(array.array("B", fh.read()),
											dtype=np.uint8).reshape(num_data, rows, cols)

	for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
									 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
		_download(base_url + filename, filename)

	train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
	train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
	test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
	test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

	return train_images, train_labels, test_images, test_labels

# Returns a [4, 28, 28] array of MNIST images.
def mnist_char_set():
	train_images, _ , _, _ = mnist_raw()
	zero_a = train_images[1]
	zero_b = train_images[21]
	one = train_images[3]
	two = train_images[5]
	three = train_images[7]
	four = train_images[2]

	char_set = jnp.stack((zero_a, zero_b, one, two, three, four), axis=0) / 255.0
	return char_set

# returns a [Batch, 28, 28, 3] array of colored mnist images.
def mnist_colored(char_set, random_key, batch_size):
	rand1, rand2 = jax.random.split(random_key, 2)

	red_color = jnp.array([1.0, 0.0, 0.0])
	blue_color = jnp.array([0.0, 0.0, 1.0])
	green_color = jnp.array([0.0, 1.0, 0.0])

	random_color = jax.random.uniform(rand1, (batch_size,)).reshape((batch_size, 1))
	color_rb = (red_color * random_color + blue_color * (0.66-random_color)) * 1.5
	color = jnp.where(random_color < 0.66, color_rb, green_color)
	
	ids = jnp.arange(6)
	probs = jnp.array([4, 4, 4, 4, 2, 1]) / 19.0
	random_ids = jax.random.choice(rand2, ids, (batch_size,), replace=True, p=probs)

	data_batch = char_set[random_ids]
	data_batch = jnp.expand_dims(data_batch, axis=3)
	data_batch = jnp.tile(data_batch, (1, 1, 1, 3))
	data_batch = data_batch * color.reshape((batch_size, 1, 1, 3))

	# Return a jax array of labels. Shape is [Batch, 7]. First four are one-hot encoding, last three are colors.
	labels_onehot = jax.nn.one_hot(random_ids, 6)
	labels_colors = color
	labels = jnp.concatenate((labels_onehot, labels_colors), axis=1)

	# FAKE DATA
	# print(data_batch.shape)
	# data_batch = data_batch.at[:, :, :, 1].set(0)
	# data_batch = data_batch.at[:, :, :, 2].set(0)
	# data_batch = data_batch.at[:, :, :, 0].set(0)
	# data_batch = data_batch.at[:, ::2, :, 0].set(1)
	# data_batch = data_batch.at[:, ::2, ::2, 0].set(0)
	# data_batch = data_batch.at[:, ::2, ::2, 1].set(1)


	return data_batch, labels

# For each image in the batch: create a grayscale image by summing the RGB channels. Find the nearest neighbor in char_set. Return the index of the nearest neighbor.
def categorize(char_set, images):
	grayscale = jnp.sum(images, axis=3)
	closest_chars = []
	closest_colors = []
	errors = []
	for img in grayscale:
		closest = jnp.inf
		closest_id = None
		for i, char in enumerate(char_set):
			# distance = jnp.linalg.norm(img - char)
			distance = jnp.sum(jnp.abs(img - char) > 0.5)
			if distance < closest:
				closest = distance
				closest_id = i
		closest_chars.append(closest_id)
		errors.append(closest)

	# Now do the same for colors. Find the average color (not including black) and find the nearest neighbor in the color set.
	color_set = jnp.array([
		[0.875, 0.0, 0.125],
		[0.625, 0.0, 0.375],
		[0.375, 0.0, 0.625],
		[0.125, 0.0, 0.875],
		[0.0, 1.0, 0.0]
	])
	avg_color = jnp.sum(images, axis=(1, 2))
	avg_color = avg_color / jnp.sum(avg_color, axis=1, keepdims=True)
	for img in avg_color:
		closest = jnp.inf
		closest_id = None
		for i, col in enumerate(color_set):
			distance = jnp.linalg.norm(img - col)
			if distance < closest:
				closest = distance
				closest_id = i
		closest_colors.append(closest_id)

	return jnp.array(closest_chars), jnp.array(closest_colors), errors


def bin(x): # X is a three-vector.
    red = jnp.floor(x[0]*8).astype('int32').clip(0, 7)
    green = jnp.floor(x[1]*8).astype('int32').clip(0, 7)
    blue = jnp.floor(x[2]*8).astype('int32').clip(0, 7)
    return red + green*8 + blue*8*8
batch_bin = jax.vmap(bin)

def bins_to_colors(bins): # Input [Batch]
    red = bins % 8
    green = (bins // 8) % 8
    blue = (bins // (8*8)) % 8
    red_axis = red.astype('float32') / 8
    green_axis = green.astype('float32') / 8
    blue_axis = blue.astype('float32') / 8
    colors = jnp.stack([red_axis, green_axis, blue_axis], axis=-1)
    return colors
batch_bins_to_colors = jax.vmap(bins_to_colors)

@jax.jit
def process_data_bins(images): # [Batch, 28, 28, 3]
	# Flatten the data, and convert to bins. Then convert back to colors.
	# Append the coordinates to the colors.
	# Return both the bins, and the augmented colors.
	x = jnp.linspace(0, 1, 28)
	y = jnp.linspace(0, 1, 28)
	xv, yv = jnp.meshgrid(x, y, indexing='ij')
	xv = xv.reshape((1, 28, 28, 1)).repeat(images.shape[0], axis=0)
	yv = yv.reshape((1, 28, 28, 1)).repeat(images.shape[0], axis=0)
	coordinates = jnp.concatenate([xv, yv], axis=-1).reshape(-1, 2)
	negative_coordinates = 1-coordinates
	is_edge = (yv == 1).reshape(-1, 1)

	bins = batch_bin(images[:,:,:,:3].reshape((-1, 3)))
	colors = batch_bins_to_colors(bins)
	colors = jnp.concatenate([colors, coordinates, negative_coordinates, is_edge], axis=-1)

	bins = bins.reshape((images.shape[0], -1))
	bins = jax.nn.one_hot(bins, 8*8*8)

	colors = colors.reshape((images.shape[0], -1, 8))
	return bins, colors

def mnist_shift(char_set, random_key, batch_size):
	data_batch, labels = mnist_colored(char_set, random_key, batch_size)
	data_batch -= 0.5 # Center data.
	return data_batch, labels

def mnist_small(char_set, random_key, batch_size):
	data_batch, labels = mnist_colored(char_set, random_key, batch_size)
	data_batch = data_batch[:, ::2, ::2]
	data_batch -= 0.5 # Center data.
	return data_batch, labels