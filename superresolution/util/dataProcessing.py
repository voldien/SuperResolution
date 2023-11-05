import concurrent.futures as cf
import imghdr
import os.path
import pathlib
import zipfile
from io import BytesIO  # for Python 3
from typing import Tuple

import PIL
import PIL.Image
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import layers
from tensorflow.python.data import Dataset


def configure_dataset_performance(ds: Dataset, use_cache: bool, cache_path: str, shuffle_size: int = 0,
								  AUTOTUNE: object = tf.data.AUTOTUNE) -> Dataset:
	if cache_path is not None:
		ds = ds.cache(filename=cache_path, name='SuperResolutionCache')
	elif use_cache:
		ds = ds.cache()
	if shuffle_size > 0:
		ds = ds.shuffle(buffer_size=shuffle_size, reshuffle_each_iteration=True)
	ds = ds.prefetch(buffer_size=AUTOTUNE)

	return ds


def load_dataset_from_directory(data_path: str, args, override_size=None, use_float16: bool = False,
								**kwargs) -> Dataset:
	# Determine if path or file.
	if os.path.isdir(data_path):
		# Set precision to be used.
		float_precision = tf.float32
		if use_float16:
			float_precision = tf.float16

		# Set import image size.
		image_size = args.image_size
		if override_size:
			image_size = override_size

		#
		color_mode = 'rgb' if args.color_channels == 3 else 'gray'
		train_ds = tf.keras.utils.image_dataset_from_directory(
			data_path,
			interpolation='bilinear',
			color_mode=color_mode,
			label_mode=None,
			crop_to_aspect_ratio=True,
			shuffle=False,
			follow_links=True,
			batch_size=None,
			image_size=image_size)

		@tf.function
		def setup_color_encoding(img, color_space):
			def preprocess_rgb2lab(tensorData):
				image = tf.cast(tensorData, float_precision)
				return tf.cast(tfio.experimental.color.rgb_to_lab(image), float_precision)

			# Convert color space encoding and normalize values.
			if color_space == 'lab':
				# Convert to LAB color and Transform [-128,128] -> [-1, 1]
				return preprocess_rgb2lab(img)
			elif color_space == 'rgb':
				# Normalize and Transform [0,1] -> [-1, 1]
				return (img * 2.0) - 1

		# Remap to [0,1]
		normalization_layer = tf.keras.layers.Rescaling(1. / 255.0)
		train_ds = train_ds.map(lambda x: normalization_layer(x))

		# Setup color space mapping.
		normalized_ds = train_ds.map(lambda x: setup_color_encoding(setup_color_encoding, args.color_space))
		# Cast data.
		normalized_ds = normalized_ds.map(lambda x: tf.cast(x, float_precision))

		return normalized_ds

		pass
	elif os.path.isfile(data_path):
		assert 0  # TODO add support


def augment_dataset(dataset: Dataset, image_crop_shape: tuple) -> Dataset:
	"""
	Augment data to minimize overfitting.
	:param dataset:
	:param image_crop_shape: Size of the final cropped result.
	:return: Augmented DataSet
	"""
	trainAug = tf.keras.Sequential([
		# Select random section of Image.
		tf.keras.layers.RandomCrop(
			image_crop_shape[0], image_crop_shape[1]),
		# Flip image around on each axis randomly.
		layers.RandomFlip("horizontal_and_vertical"),
		# Random Zoom.
		layers.RandomZoom(
			height_factor=(-0.05, -0.15),
			width_factor=(-0.05, -0.15)),
		# Random Rotation.
		layers.RandomRotation(0.65)
	])

	def AgumentFunc(x):
		aX = trainAug(x)
		return aX

	#
	dataset = (
		dataset
		.map(AgumentFunc))

	return dataset


def dataset_super_resolution(dataset: Dataset, input_size, output_size) -> Dataset:
	"""
	Perform Super Resolution Data and Expected Data to Correct Size. For providing
	the model with corrected sized Data.
	:param dataset: Valid DataSet
	:param input_size: Input Image Size
	:param output_size: Output Image Size.
	:return: DataSet with both Dat and Expected Data.
	"""

	def DownScaleLayer(data):
		"""

		:param data:
		:return:
		"""
		downScale = tf.keras.Sequential([
			layers.Resizing(
				input_size[0],
				input_size[1],
				interpolation='bilinear',
				crop_to_aspect_ratio=False
			)])

		# Create a copy to prevent augmentation be done twice separately.
		expected = tf.identity(data)

		# Remap from [-1, 1] to [0,1]
		data = (data + 1.0) * 0.5
		# Downscale Input Data
		data = downScale(data)
		# Remap from [0, 1] to [-1,1]
		data = (2.0 * data) - 1.0

		return data, expected

	DownScaledDataSet = (
		dataset
		.map(DownScaleLayer,
			 num_parallel_calls=tf.data.AUTOTUNE)
	)

	return DownScaledDataSet


def split_dataset(dataset: Dataset, train_size: float) -> Tuple[Dataset, Dataset]:
	"""

	:param dataset:
	:param train_size:
	:return:
	"""
	nrBatches = len(dataset)

	#
	offset_skip = int(train_size * nrBatches)
	validation_nr_batchs = int((1.0 - train_size) * nrBatches)

	#
	train_ds = dataset.take(offset_skip)
	validation_ds = dataset.skip(offset_skip).take(validation_nr_batchs)

	return train_ds, validation_ds


# TODO
def processImageDataset(train_images):
	# Do per section of the
	for i, image in enumerate(train_images.astype('float32')):
		train_images[i] = train_images[i] / 255.0
	return train_images


# TODO make the filter a lamba method
def loadImageDataSubSet(path, subset, resize=(128, 128), filter=(".jpg", ".JPG", ".png", ".png")):
	images = []
	_n = int(len(subset))
	with zipfile.ZipFile(path, 'r') as zip:
		for i in range(_n):
			file_in_zip = subset[i]

			if pathlib.Path(file_in_zip).suffix in filter:
				try:
					data = zip.read(file_in_zip)
					stream = BytesIO(data)
					if imghdr.what(stream) is not None:
						image = PIL.Image.open(stream)
						image = image.resize(resize, PIL.Image.BILINEAR)
						images.append(np.asarray(image))
					stream.close()
				except Exception as exc:
					print('{0} generated an exception: {1}'.format(
						file_in_zip, exc))
	return images


def load_image_data(pool, path, size):
	future_to_image = []
	with zipfile.ZipFile(path, 'r') as zip:
		zlist = zip.namelist()
		nr_chunks = 32
		chunk_size = int(len(zlist) / nr_chunks)
		for i in range(nr_chunks):
			subset = zlist[chunk_size * i: chunk_size * (i + 1)]
			task = pool.submit(loadImageDataSubSet, path, subset, size)
			future_to_image.append(task)
	return future_to_image


def loadImagedataSet(path, filter=None, ProcessOverride=None, size=(128, 128)):
	future_to_image = []
	total_data = []
	with cf.ProcessPoolExecutor() as pool:
		for f in load_image_data(pool, path, size):
			future_to_image.append(f)
			for future in cf.as_completed(set(future_to_image)):
				try:
					data = future.result()
					for x in data:
						total_data.append(x)
				except Exception as exc:
					print('{0} generated an exception: {1}'.format("url", exc))
				del data
	return (np.asarray(total_data), None), (None, None)


def loadImageDataSet(paths, filter=None, ProcessOverride=None, size=(128, 128)):
	image_sets = []
	for path in paths:
		(image, _0), (_1, _2) = loadImagedataSet(
			path, filter, ProcessOverride, size)
		image_sets.append(image)
	return (np.concatenate((x for x in image_sets)), None), (None, None)
