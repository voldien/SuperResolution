import argparse
from models import create_activation

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class SuperResolutionModelCNN(ModelBase):
	def __init__(self):
		self.possible_upscale = [2, 4]
		self.parser = argparse.ArgumentParser(add_help=False)

		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.000001,
								 required=False,
								 help='Set the L1 Regularization applied.')


	def load_argument(self) -> argparse.ArgumentParser:
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		scale_factor: int = int(output_shape[0] / input_shape[0])
		scale_factor: int = int(output_shape[1] / input_shape[1])

		if scale_factor not in self.possible_upscale and scale_factor not in self.possible_upscale:
			raise ValueError("Invalid upscale")

		# Model Construct Parameters.
		regularization: float = kwargs.get("regularization", 0.000001)  #
		upscale_mode: int = scale_factor  #
		num_input_filters: int = kwargs.get("input_filters", 64)  #

		#
		return create_cnn_model(input_shape=input_shape,
								output_shape=output_shape, input_filter_size=num_input_filters, regularization=regularization,
								upscale_mode=upscale_mode,
								kernel_activation='relu')

	def get_name(self):
		return "Basic SuperResolution"


def get_model_interface() -> ModelBase:
	return SuperResolutionModelCNN()


def create_cnn_model(input_shape: tuple, output_shape: tuple, input_filter_size: int, regularization: float,
					 upscale_mode: int,
					 kernel_activation: str):

	use_batch_norm: bool = True
	use_bias: bool = True
	num_conv_block: int = 3

	output_width, output_height, output_channels = output_shape

	x = input_layer = layers.Input(shape=input_shape)

	for i in range(0, int(upscale_mode / 2)):
		x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

	# Convolutional block
	for _ in range(0, num_conv_block):
		filter_size = input_filter_size << i
		x = layers.Conv2D(filters=filter_size, kernel_size=(3, 3), strides=1, padding='same', use_bias=use_bias,
						  kernel_initializer=tf.keras.initializers.HeNormal())(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

	# Output to 3 channel output.
	x = layers.Conv2DTranspose(filters=output_channels, kernel_size=(9, 9), strides=(
		1, 1), padding='same', use_bias=use_bias, kernel_initializer=tf.keras.initializers.HeNormal(),
		bias_initializer=tf.keras.initializers.HeNormal())(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=regularization, l2=0)(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	return keras.Model(inputs=input_layer, outputs=x, name="cnnsr")
