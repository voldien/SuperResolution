import argparse

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class EDSRSuperResolutionModel(ModelBase):
	def __init__(self):
		self.parser = argparse.ArgumentParser(add_help=True, prog="", description="")

		#
		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.001,
								 required=False,
								 help='Set the L1 Regularization applied.')

		#
		self.parser.add_argument('--upscale-mode', dest='upscale_mode',
								 type=int,
								 choices=[2, 4],
								 default=2,
								 required=False,
								 help='Upscale Mode')
		#
		self.parser.add_argument('--use-resnet', type=bool, default=False, dest='use_resnet',
								 help='Set the number of passes that the training set will be trained against.')

		#
		self.parser.add_argument('--edsr_filters', type=int, default=192, dest='edsr_filters',
								 help='')

	def load_argument(self) -> argparse.ArgumentParser:
		#
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		# Model constructor parameters.
		regularization = kwargs.get("regularization", 0.00001)  #
		upscale_mode = kwargs.get("upscale_mode", 2)  #
		num_input_filters = kwargs.get("edsr_filters", 192)  #
		use_resnet = kwargs.get("use_resnet", True)  #

		#
		return create_model(input_shape=input_shape,
							output_shape=output_shape, scale=upscale_mode,
							num_filters=num_input_filters, regularization=regularization)

	def get_name(self):
		return "SuperResolution - EDSR"


def get_model_interface() -> ModelBase:
	return EDSRSuperResolutionModel()


def create_model(input_shape, output_shape, scale, num_filters=64, num_res_blocks=8, res_block_scaling=None,
				 regularization=0.00001):
	"""Creates an EDSR model."""

	output_width, output_height, output_channels = output_shape

	x_in = layers.Input(shape=input_shape)

	#
	x = b = layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x_in)
	for i in range(num_res_blocks):
		b = res_block(b, num_filters, res_block_scaling)

	#
	b = layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(b)
	x = layers.Add()([x, b])

	#
	x = upsample(x, scale, num_filters)
	x = layers.Conv2D(filters=output_channels, kernel_size=(3, 3), padding='same')(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=regularization, l2=0)(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	return keras.Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
	"""Creates an EDSR residual block."""
	x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x_in)
	x = layers.Conv2D(filters, 3, padding='same')(x)
	if scaling:
		x = layers.Lambda(lambda t: t * scaling)(x)
	x = layers.Add()([x_in, x])
	return x


def upsample(x, scale, num_filters):
	def upsample_1(x, factor, **kwargs):
		"""Sub-pixel convolution."""
		x = layers.Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
		return tf.nn.depth_to_space(x, 2)

	if scale == 2:
		x = upsample_1(x, 2, name='conv2d_1_scale_2')
	elif scale == 3:
		x = upsample_1(x, 3, name='conv2d_1_scale_3')
	elif scale == 4:
		x = upsample_1(x, 2, name='conv2d_1_scale_2')
		x = upsample_1(x, 2, name='conv2d_2_scale_2')

	return x
