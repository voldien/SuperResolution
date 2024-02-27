import argparse

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class DCPostSuperResolutionModel(ModelBase):
	def __init__(self):
		pass

	def load_argument(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(add_help=False, prog="Post Basic SuperResolution",
										 description="Basic Deep Convolutional Super Resolution")

		group = parser.add_argument_group(self.get_name())

		#
		group.add_argument('--regularization', dest='regularization', required=False,
						   type=float,
						   default=0.0001,
						   help='Set the L1 Regularization applied.')
		#
		return parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		return create_post_super_resolution(input_shape=input_shape, output_shape=output_shape)

	def get_name(self):
		return "post super"


def get_model_interface() -> ModelBase:
	return DCPostSuperResolutionModel()


def create_post_super_resolution(input_shape: tuple, output_shape: tuple):
	batch_norm: bool = True
	use_bias: bool = True

	output_width, output_height, output_channels = output_shape

	x = input_layer = layers.Input(shape=input_shape)

	number_layers = 3

	x = layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(
		1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
	upscale = x

	for i in range(0, number_layers):
		filter_size = 2 ** (i + 6)
		filter_size = min(filter_size, 1024)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), strides=1, padding='same',
						  kernel_initializer=tf.keras.initializers.HeNormal())(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1,
						  kernel_initializer=tf.keras.initializers.HeNormal())(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1,
						  kernel_initializer=tf.keras.initializers.HeNormal())(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = layers.ReLU(dtype='float32')(x)

	# x = tf.nn.depth_to_space(x, 2)

	x = layers.Conv2D(filters=3, kernel_size=(4, 4), strides=(
		1, 1), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = layers.Activation('tanh')(x)  # TODO: determine

	x = layers.add([x, upscale])
	x = layers.Activation('tanh')(x)
	# x = layers.ActivityRegularization(l1=0.0001)(x)

	conv_autoencoder = keras.Model(inputs=input_layer, outputs=x)
	return conv_autoencoder
