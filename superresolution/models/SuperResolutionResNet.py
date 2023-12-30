import argparse

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class ResNetSuperResolutionModel(ModelBase):
	def __init__(self):
		self.parser = argparse.ArgumentParser(add_help=True, prog="", description="")

		self.parser.add_argument('--use-resnet', type=bool, default=False, dest='use_resnet',
								 help='Set the number of passes that the training set will be trained against.')

		self.parser.add_argument('--override-latentspace-size', dest='generate_latentspace',
								 default=False,
								 help='', type=bool)
		#
		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.001,
								 help='Set the L1 Regularization applied.')

		self.parser.add_argument('--upscale-mode', dest='upscale_mode',
								 type=str,
								 choices=[''],
								 default='',
								 help='Set the L1 Regularization applied.')

	def load_argument(self) -> argparse.ArgumentParser:
		"""Load in the file for extracting text."""

		#
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		"""Extract text from the currently loaded file."""
		#
		# parser_result = self.parser.parse_known_args(sys.argv[1:])

		# Model constructor parameters.
		regularization: float = kwargs.get("regularization", 0.00001)  #
		upscale_mode: int = kwargs.get("upscale_mode", 2)  #
		num_input_filters: int = kwargs.get("edsr_filters", 256)  #
		num_res_blocks = kwargs.get("num_res_blocks", 8)  #

		#
		return create_simple_model(input_shape=input_shape,
								   output_shape=output_shape, num_res_blocks=num_res_blocks, regularization=regularization)

	def get_name(self):
		return "basic super"

def create_activation(activation):
	if activation == "leaky_relu":
		return layers.LeakyReLU(alpha=0.2, dtype='float32')
	elif activation == "relu":
		return layers.ReLU(dtype='float32')
	elif activation == "sigmoid":
		return layers.Activation(activation='sigmoid', dtype='float32')
	else:
		assert "Should never be reached"

def get_model_interface() -> ModelBase:
	return ResNetSuperResolutionModel()


def residual_block(input, filters=64, use_batch_norm=False):
	start_ref = input

	init = tf.keras.initializers.HeNormal()

	x = layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(input)
	if use_batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)
	x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(input)

	x = layers.add([start_ref, x])

	return x


def create_simple_model(input_shape, output_shape, num_res_blocks=8, regularization=0.00001):
	batch_norm: bool = True
	use_bias: bool = True

	init = tf.keras.initializers.HeNormal()
	output_width, output_height, output_channels = output_shape

	input = layers.Input(shape=input_shape)
	number_layers = 2

	x = input
	for i in range(0, num_res_blocks):
		x = residual_block(input=x, filters=64, use_batch_norm= False)

	for i in range(0, number_layers):
		filter_size = 2 ** (i + 7)
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
	x = layers.Conv2DTranspose(filters=3, kernel_size=(9, 9), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=regularization)(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder