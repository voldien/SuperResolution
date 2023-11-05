import argparse

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class DCSuperResolutionModel(ModelBase):
	def __init__(self):
		self.parser = argparse.ArgumentParser(add_help=True, prog="Basic SuperResolution",
											  description="Basic Deep Convolutional Super Resolution")
		#
		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.000001,
								 required=False,
								 help='Set the L1 Regularization applied.')

		#
		self.parser.add_argument('--upscale-mode', dest='upscale_mode',
								 type=int,
								 choices=[2, 4],
								 default=2,
								 required=False,
								 help='Upscale Mode')

	def load_argument(self) -> argparse.ArgumentParser:
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		# Model Construct Parameters.
		regularization = kwargs.get("regularization", 0.000001)  #
		upscale_mode = kwargs.get("upscale_mode", 2)  #

		#
		return create_simple_model(input_shape=input_shape,
								   output_shape=output_shape, regularization=regularization, upscale_mode=upscale_mode)

	def get_name(self):
		return "Basic SuperResolution"


def get_model_interface() -> ModelBase:
	return DCSuperResolutionModel()


# Create interface object or similar.
def create_simple_model(input_shape: tuple, output_shape: tuple, regularization: float, upscale_mode: int):
	batch_norm: bool = True
	use_bias: bool = True

	init = tf.keras.initializers.HeNormal()
	output_width, output_height, output_channels = output_shape

	input = layers.Input(shape=input_shape)

	for i in range(0, int(upscale_mode / 2)):
		nrfilters = output_width
		#
		x = layers.Conv2D(filters=nrfilters, kernel_size=(9, 9), strides=1, padding='same',
						  use_bias=use_bias,
						  kernel_initializer=init, bias_initializer=init)(input)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		#
		x = layers.Conv2D(filters=nrfilters / 2, kernel_size=(4, 4), strides=1, padding='same',
						  use_bias=use_bias,
						  kernel_initializer=init, bias_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		#
		x = layers.Conv2D(filters=nrfilters / 4, kernel_size=(3, 3), strides=1, padding='same',
						  use_bias=use_bias,
						  kernel_initializer=init, bias_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		# Upscale -
		x = layers.Conv2DTranspose(filters=output_width, kernel_size=(5, 5), strides=(
			2, 2), use_bias=use_bias, padding='same', kernel_initializer=init, bias_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		#
		x = layers.Conv2D(filters=nrfilters, kernel_size=(4, 4), strides=1, padding='same',
						  use_bias=use_bias,
						  kernel_initializer=init, bias_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

	# Output to 3 channel output.
	x = layers.Conv2DTranspose(filters=output_channels, kernel_size=(9, 9), strides=(
		1, 1), padding='same', use_bias=use_bias, kernel_initializer=init, bias_initializer=init)(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=regularization, l2=0)(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder


def create_simple_model4(input_shape, output_shape):
	batch_norm: bool = True
	use_bias: bool = True

	init = tf.keras.initializers.Orthogonal()

	input = layers.Input(shape=input_shape)
	number_layers = 3

	x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
					  kernel_initializer=init)(input)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)

	for i in range(0, number_layers):
		filter_size = 2 ** (i + 6)
		filter_size = min(filter_size, 1024)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

	x = tf.nn.depth_to_space(x, 2)

	# Upscale.
	x = layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=0.0001)(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder


def create_simple_model3(input_shape, output_shape):
	batch_norm = False

	init = tf.keras.initializers.Orthogonal()

	input = layers.Input(shape=input_shape)
	number_layers = 3

	x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
					  kernel_initializer=init)(input)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)

	for i in range(0, number_layers):
		filter_size = 2 ** (i + 5)
		filter_size = min(filter_size, 1024)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

	# x = tf.nn.depth_to_space(x, 2)

	# x = layers.Conv2DTranspose(filters=filter_size, kernel_size=(3, 3), strides=(2, 2), use_bias=False,
	#						   padding='same',
	#						   kernel_initializer=init)(x)

	# for i in range(0, 1):
	#	filter_size = 2 ** (7)
	#	filter_size = min(filter_size, 1024)
	#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)
	#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)
	#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)

	x = layers.Conv2D(filters=3, kernel_size=(9, 9), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)
	# x = layers.ActivityRegularization(l1=0.0001)(x)

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder
