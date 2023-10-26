import argparse
import sys
from core import ModelBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DCSuperResolutionModel(ModelBase):
	def __init__(self):
		self.parser = argparse.ArgumentParser(add_help=True, prog="Basic SuperResolution", description="Basic Deep Convolutional Super Resolution")
		#
		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.001,
								 help='Set the L1 Regularization applied.')

		#
		self.parser.add_argument('--upscale-mode', dest='upscale_mode',
								 type=int,
								 choices=[2,4,],
								 default=2,
								 help='Upscale Mode')

		#
		self.parser.add_argument('--loss-fn', dest='loss_fn',
								 default='mse',
								 choices=['mses'],
								 help='.', type=str)

	def load_argument(self) -> argparse.ArgumentParser:
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		#
		parser_result = self.parser.parse_args(sys.argv[1:])

		#
		return create_simple_model(input_shape=input_shape,
								   output_shape=output_shape, regularization=parser_result.regularization, upscale_mode=parser_result.upscale_mode)

	def get_name(self):
		return "Basic SuperResolution"


def get_model_interface() -> ModelBase:
	return DCSuperResolutionModel()


# Create interface object or similar.
def create_simple_model(input_shape, output_shape, regularization=0.00000, upscale_mode=2):
	batch_norm: bool = True
	use_bias: bool = True

	init = tf.keras.initializers.HeNormal()
	output_width, output_height, output_channels = output_shape

	input = layers.Input(shape=input_shape)

	#
	x = layers.Conv2D(output_width, kernel_size=(9, 9), strides=1, padding='same',
					  use_bias=use_bias,
					  kernel_initializer=init)(input)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)

	#
	x = layers.Conv2D(output_width / 2, kernel_size=(3, 3), strides=1, padding='same',
					  use_bias=use_bias,
					  kernel_initializer=init)(x)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)

	# Upscale - 
	x = layers.Conv2DTranspose(filters=output_width / 2, kernel_size=(5, 5), strides=(
		2, 2), use_bias=use_bias, padding='same', kernel_initializer=init)(x)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)

	#
	x = layers.Conv2DTranspose(filters=output_channels, kernel_size=(5, 5), strides=(
		1, 1), padding='same', use_bias=use_bias, kernel_initializer=init)(x)
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

		x = layers.Conv2D(filter_size, kernel_size=(3, 3),  strides=1, padding='same', kernel_initializer=init)(x)
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
