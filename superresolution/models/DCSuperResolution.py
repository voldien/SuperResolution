import argparse
from models import create_activation

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
		regularization: float = kwargs.get("regularization", 0.000001)  #
		upscale_mode: int = kwargs.get("upscale_mode", 2)  #

		#
		return create_simple_model(input_shape=input_shape,
								   output_shape=output_shape, regularization=regularization, upscale_mode=upscale_mode,
								   kernel_activation='relu')

	def get_name(self):
		return "Basic SuperResolution"


def get_model_interface() -> ModelBase:
	return DCSuperResolutionModel()


def create_simple_model(input_shape: tuple, output_shape: tuple, regularization: float, upscale_mode: int,
						kernel_activation: str):
	use_batch_norm: bool = True
	use_bias: bool = True

	output_width, output_height, output_channels = output_shape

	x = input_layer = layers.Input(shape=input_shape)

	for i in range(0, int(upscale_mode / 2)):
		nrfilters = output_width
		#
		x = layers.Conv2D(filters=nrfilters, kernel_size=(9, 9), strides=1, padding='same',
						  use_bias=use_bias,
						  kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal())(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		#
		x = layers.Conv2D(filters=nrfilters / 2, kernel_size=(4, 4), strides=1, padding='same',
						  use_bias=use_bias,
						  kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal())(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		#
		x = layers.Conv2D(filters=nrfilters / 4, kernel_size=(3, 3), strides=1, padding='same',
						  use_bias=use_bias,
						  kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal())(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		# Upscale -
		x = layers.Conv2DTranspose(filters=output_width, kernel_size=(5, 5), strides=(
			2, 2), use_bias=use_bias, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal())(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		#
		x = layers.Conv2D(filters=nrfilters, kernel_size=(4, 4), strides=1, padding='same',
						  use_bias=use_bias,
						  kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal())(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

	# Output to 3 channel output.
	x = layers.Conv2DTranspose(filters=output_channels, kernel_size=(9, 9), strides=(
		1, 1), padding='same', use_bias=use_bias, kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal())(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=regularization, l2=0)(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	return keras.Model(inputs=input_layer, outputs=x)
