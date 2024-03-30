import argparse

from models import create_activation

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class VDSRSuperResolutionModel(ModelBase):
	def __init__(self):
		self.possible_upscale = [2, 4]

		
		self.parser = argparse.ArgumentParser(add_help=False)
		#
		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.00001,
								 help='Set the L1 Regularization applied.')
		self.parser.add_argument('--upscale-mode', dest='upscale_mode',
								 type=int,
								 choices=[2, 4],
								 default=2,
								 required=False,
								 help='Upscale Mode')

	def load_argument(self) -> argparse.ArgumentParser:
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		scale_factor: int = int(output_shape[0] / input_shape[0])
		scale_factor: int = int(output_shape[1] / input_shape[1])

		if scale_factor not in self.possible_upscale and scale_factor not in self.possible_upscale:
			raise ValueError("Invalid upscale")
		
		# parser_result = self.parser.parse_known_args(sys.argv[1:])
		# Model constructor parameters.

		regularization: float = kwargs.get("regularization", 0.00001)  #
		upscale_mode: int = scale_factor  #
		num_input_filters: int = kwargs.get("edsr_filters", 128)  #

		#
		return create_vdsr_model(input_shape=input_shape,
								 output_shape=output_shape,
								 filters=num_input_filters,
								 upscale_mode=upscale_mode,
								 regularization=regularization,
								 kernel_activation='relu')

	def get_name(self):
		return "VDSR Super Resolution"


def get_model_interface() -> ModelBase:
	return VDSRSuperResolutionModel()


def create_vdsr_model(input_shape: tuple, output_shape: tuple, filters: int, kernel_activation: str,
					  upscale_mode: int = 2, regularization: float = 0.00001):
	use_batch_norm: bool = True
	use_bias: bool = True

	number_layers: int = 2
	num_conv_block: int = 2

	output_width, output_height, output_channels = output_shape

	x = input_layer = layers.Input(shape=input_shape)

	# Upscale to fit the end upscaled version.
	upscale = x
	for _ in range(0, int(upscale_mode / 2)):
		# TODO add upscale modes.
		upscale = layers.Conv2DTranspose(filters=(filters << (number_layers - 1)), kernel_size=(4, 4), strides=(
			2, 2), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(upscale)

	filter_size: int = 0
	for i in range(0, number_layers):
		filter_size = filters << i
		filter_size = min(filter_size, 1024)

		for _ in range(0, num_conv_block):
			x = layers.Conv2D(filters=filter_size, kernel_size=(3, 3), strides=1, padding='same', use_bias=use_bias,
							  kernel_initializer=tf.keras.initializers.HeNormal())(x)
			if use_batch_norm:
				x = layers.BatchNormalization(dtype='float32')(x)
			x = create_activation(kernel_activation)(x)

	# Upscale
	for _ in range(0, int(upscale_mode / 2)):
		assert filter_size != 0

		x = layers.Conv2DTranspose(filters=filter_size, kernel_size=(4, 4), strides=(
			2, 2), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)

	# Combine orignal with end.
	x = layers.add([x, upscale])

	# Output
	x = layers.Conv2D(filters=output_channels, kernel_size=(4, 4), strides=(1, 1),
					  padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=regularization, l2=0)(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	return keras.Model(inputs=input_layer, outputs=x, name="vdsr")
