import argparse
from models import create_activation

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class AESuperResolutionModel(ModelBase):
	def __init__(self):
		self.parser = argparse.ArgumentParser(add_help=False)

		self.parser.add_argument('--use-resnet', type=bool, default=False, dest='use_resnet',
								 help='Set the number of passes that the training set will be trained against.')

		self.parser.add_argument('--override-latentspace-size', dest='generate_latentspace',
								 default=False,
								 help='override latent space size', type=bool)
		#
		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.00001,
								 help='Set the L1 Regularization applied.')
		
		#
		self.parser.add_argument('--filters', type=int, default=64, dest='filters',
								 help='Set Filter Count')

	def load_argument(self) -> argparse.ArgumentParser:
		return self.parser

	def create_model(self, input_shape: tuple, output_shape: tuple, **kwargs) -> keras.Model:
		scale_width_factor, scale_height_factor = self.compute_upscale_mode(input_shape, output_shape)

		if scale_width_factor not in self.possible_upscale and scale_height_factor not in self.possible_upscale:
			raise ValueError("Invalid upscale")


		regularization: float = kwargs.get("regularization", 0.00001)  #
		upscale_mode: int = scale_width_factor #
		num_input_filters: int = kwargs.get("filters", 64)  #
		use_resnet: bool = kwargs.get("use_resnet", True)  #

		#
		return create_dscr_autoencoder_model(input_shape=input_shape,
											 output_shape=output_shape, use_resnet=use_resnet,
											 input_filters=num_input_filters,
											 regularization=regularization, use_upresize=False,
											 kernel_activation='relu')

	def get_name(self):
		return "Auto Encoder Super Resolution"


def get_model_interface() -> ModelBase:
	return AESuperResolutionModel()


def create_dscr_autoencoder_model(input_shape: tuple, output_shape: tuple, use_resnet: bool, regularization: float,
								  use_upresize: bool,
								  kernel_activation: str, input_filters: int):
	use_batch_norm: bool = True
	use_bias: bool = True

	output_width, output_height, output_channels = output_shape

	x = input_layer = layers.Input(shape=input_shape)
	number_layers: int = 2
	num_conv_block: int = 2
	offset_degre: int = 6

	x = layers.Conv2D(filters=input_filters, kernel_size=(3, 3), strides=1, padding='same',
					  kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
	if use_batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)

	last_sum_layer = x

	for i in range(0, number_layers):
		filter_size = input_filters << i
		filter_size = min(filter_size, 1024)

		for j in range(0, num_conv_block):
			x = layers.Conv2D(filters=filter_size, kernel_size=(3, 3), strides=1, padding='same',
							  kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
			if use_batch_norm:
				x = layers.BatchNormalization(dtype='float32')(x)
			x = create_activation(kernel_activation)(x)

		# Downscale layer
		x = layers.Conv2D(filters=filter_size, kernel_size=(3, 3), strides=2, padding='same',
						  kernel_initializer=tf.keras.initializers.GlorotUniform())(
			x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		attach_layer = x
		if use_resnet:
			if last_sum_layer is not None:
				last_sum_layer = layers.Conv2D(filters=filter_size, kernel_size=(1, 1),
											   kernel_initializer=tf.keras.initializers.GlorotUniform(),
											   strides=(2, 2))(last_sum_layer)
				x = layers.add([attach_layer, last_sum_layer])

			last_sum_layer = x
			x = create_activation(kernel_activation)(x)

	connect_conv_shape = x.shape

	x = layers.Flatten(name="latentspace")(x)
	#
	x = layers.ActivityRegularization(l1=regularization, l2=0)(x)

	x = layers.Reshape(target_shape=(
		connect_conv_shape[1], connect_conv_shape[2], connect_conv_shape[3]))(x)

	last_sum_layer = x
	for i in range(0, number_layers + 1):
		filter_size = input_filters << (number_layers - i + 1)
		filter_size = min(filter_size, 1024)

		if use_upresize:
			x = layers.UpSampling2D(size=(2, 2))(x)
		else:
			x = layers.Conv2DTranspose(filters=filter_size, kernel_size=(
				4, 4), kernel_initializer=tf.keras.initializers.GlorotUniform(), strides=(2, 2), padding='same')(x)

		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		for _ in range(0, num_conv_block):
			x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1,
							  kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
			if use_batch_norm:
				x = layers.BatchNormalization(dtype='float32')(x)
			x = create_activation(kernel_activation)(x)

		attach_layer = x
		if use_resnet:
			if last_sum_layer is not None:
				last_sum_layer = layers.Conv2DTranspose(filters=filter_size, kernel_size=(
					1, 1), kernel_initializer=tf.keras.initializers.GlorotUniform(), strides=(2, 2))(last_sum_layer)
				x = layers.add([attach_layer, last_sum_layer])
			last_sum_layer = x
			x = create_activation(kernel_activation)(x)

	x = layers.Conv2D(filters=output_channels, kernel_size=(9, 9), strides=(
		1, 1), padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
	x = layers.Activation('tanh')(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	return keras.Model(inputs=input_layer, outputs=x, name="aesr")
