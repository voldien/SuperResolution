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
								  prefetch_buffer_size: object = tf.data.AUTOTUNE) -> Dataset:
	if cache_path is not None:
		ds = ds.cache(filename=cache_path, name='SuperResolutionCache')
	elif use_cache:
		ds = ds.cache()
	if shuffle_size > 0:
		ds = ds.shuffle(buffer_size=shuffle_size, reshuffle_each_iteration=True)

	ds = ds.prefetch(buffer_size=prefetch_buffer_size)

	return ds


def load_dataset_from_directory(data_path: str, args: dict, override_size=None, use_float16: bool = False,
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

		split: float = 0
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

		normalization_layer = tf.keras.layers.Rescaling(1. / 255.0)

		@tf.function
		def setup_color_encoding(img, color_space: str):
			# TODO relocate to its own method.
			@tf.function
			def preprocess_rgb2lab(tensor_data):

				image = tf.cast(tensor_data, float_precision)

				return tf.cast(tfio.experimental.color.rgb_to_lab(image), float_precision)

			# Convert color space encoding and normalize values.
			if color_space == 'lab':
				# Convert to LAB color and Transform [0, 255] -> [-128,128] -> [-1, 1]
				return preprocess_rgb2lab(normalization_layer(img)) * (1.0 / 128.0)
			elif color_space == 'rgb':
				# Normalize and Transform [0, 255] -> [0,1] -> [-1, 1]
				return (normalization_layer(img) * 2.0) - 1.0
			assert 0

		# Setup color space mapping.
		normalized_ds = train_ds.map(lambda x: setup_color_encoding(x, args.color_space))

		# Cast data.
		normalized_ds = normalized_ds.map(lambda x: tf.cast(x, float_precision))

		return normalized_ds

	elif os.path.isfile(data_path):
		raise NotImplementedError("Not supported to add file.")


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
			height_factor=(-0.05, 0.05),
			width_factor=(-0.05, 0.05),
			fill_mode='reflect',
			interpolation='bilinear'),
		# Random Rotation.
		layers.RandomRotation(factor=0.65,
							  fill_mode='reflect',
							  interpolation='bilinear')
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
	nr_batches = len(dataset)

	#
	offset_skip = int(train_size * nr_batches)
	validation_nr_batches = int((1.0 - train_size) * nr_batches)

	#
	train_ds = dataset.take(offset_skip)
	validation_ds = dataset.skip(offset_skip).take(validation_nr_batches)

	return train_ds, validation_ds


# TODO
def processImageDataset(train_images):
	# Do per section of the
	for i, image in enumerate(train_images.astype('float32')):
		train_images[i] = train_images[i] / 255.0
	return train_images


# TODO make the filter a lamba method
def loadImageDataSubSet(path, subset, resize=(128, 128), filters=(".jpg", ".JPG", ".png", ".png")):
	images = []
	_n = int(len(subset))
	with zipfile.ZipFile(path, 'r') as zip_io:
		for i in range(_n):
			file_in_zip = subset[i]

			if pathlib.Path(file_in_zip).suffix in filters:
				try:
					data = zip_io.read(file_in_zip)
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
	with zipfile.ZipFile(path, 'r') as zip_io:
		zlist = zip_io.namelist()
		nr_chunks = 32
		chunk_size = int(len(zlist) / nr_chunks)
		for i in range(nr_chunks):
			subset = zlist[chunk_size * i: chunk_size * (i + 1)]
			task = pool.submit(loadImageDataSubSet, path, subset, size)
			future_to_image.append(task)
	return future_to_image


def loadImagedataSet(path, filter_ext=None, ProcessOverride=None, size=(128, 128)):
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


def loadImageDataSet(paths, filters=None, ProcessOverride=None, size=(128, 128)):
	image_sets = []
	for path in paths:
		(image, _0), (_1, _2) = loadImagedataSet(
			path, filters, ProcessOverride, size)
		image_sets.append(image)
	return (np.concatenate((x for x in image_sets)), None), (None, None)
