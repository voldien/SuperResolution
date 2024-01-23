import argparse

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class EDSRSuperResolutionModel(ModelBase):
	def __init__(self):
		self.parser = argparse.ArgumentParser(add_help=False)

		#
		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.00001,
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
		self.parser.add_argument('--edsr-filters', type=int, default=256, dest='edsr_filters',
								 help='')

	def load_argument(self) -> argparse.ArgumentParser:
		#
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		# Model constructor parameters.
		regularization: float = kwargs.get("regularization", 0.00001)  #
		upscale_mode: int = kwargs.get("upscale_mode", 2)  #
		num_input_filters: int = kwargs.get("edsr_filters", 192)  #

		#
		return create_edsr_model(input_shape=input_shape,
								 output_shape=output_shape, scale=upscale_mode,
								 num_filters=num_input_filters, regularization=regularization)

	def get_name(self):
		return "SuperResolution - EDSR - Enchanced Deep Super Resolution"


def get_model_interface() -> ModelBase:
	return EDSRSuperResolutionModel()


def create_edsr_model(input_shape: tuple, output_shape: tuple, scale: int, num_filters: int = 64,
					  num_res_blocks: int = 8, res_block_scaling: int = None,
					  regularization: float = 0.00001):
	"""Creates an EDSR model."""

	output_width, output_height, output_channels = output_shape
	init = tf.keras.initializers.GlorotUniform()

	x_in = layers.Input(shape=input_shape)

	#
	x = _res_block = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', kernel_initializer=init)(
		x_in)
	for _ in range(num_res_blocks):
		_res_block = res_block(_res_block, num_filters, res_block_scaling)

	#
	_res_block = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', kernel_initializer=init)(
		_res_block)
	x = layers.Add()([x, _res_block])

	#
	x = upsample(x, scale, num_filters)

	# Output layer.
	x = layers.Conv2D(filters=output_channels, kernel_size=(3, 3), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=regularization, l2=0)(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	return keras.Model(x_in, x, name="edsr")


def res_block(x_in, filters: int, scaling):
	init = tf.keras.initializers.GlorotUniform()

	"""Creates an EDSR residual block."""
	x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer=init)(x_in)
	x = layers.ReLU(dtype='float32')(x)
	x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer=init)(x)

	if scaling:
		x = layers.Lambda(lambda t: t * scaling)(x)
	x = layers.Add()([x_in, x])
	return x


def upsample(x, scale: int, num_filters: int):
	def upsample_1(input_layer, factor, **kwargs):
		"""Sub-pixel convolution."""
		x_ = layers.Conv2D(filters=num_filters * (factor ** 2), kernel_size=(3, 3), padding='same', **kwargs)(input_layer)
		return tf.nn.depth_to_space(x_, 2)

	if scale == 2:
		x = upsample_1(x, 2, name='conv2d_1_scale_2')
	elif scale == 3:
		x = upsample_1(x, 3, name='conv2d_1_scale_3')
	elif scale == 4:
		x = upsample_1(x, 2, name='conv2d_1_scale_2')
		x = upsample_1(x, 2, name='conv2d_2_scale_2')

	return x
