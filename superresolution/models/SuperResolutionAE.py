import argparse
from models import create_activation

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers


class AESuperResolutionModel(ModelBase):
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
								 default=0.00001,
								 help='Set the L1 Regularization applied.')

	def load_argument(self) -> argparse.ArgumentParser:
		return self.parser

	def create_model(self, input_shape: tuple, output_shape: tuple, **kwargs) -> keras.Model:
		#
		# parser_result = self.parser.parse_known_args(sys.argv[1:])
		# Model constructor parameters.
		regularization: float = kwargs.get("regularization", 0.00001)  #
		upscale_mode: int = kwargs.get("upscale_mode", 2)  #
		num_input_filters: int = kwargs.get("edsr_filters", 256)  #
		use_resnet: bool = kwargs.get("use_resnet", True)  #

		#
		return create_dscr_auto_encoder_model(input_shape=input_shape,
											  output_shape=output_shape, use_resnet=use_resnet,
											  regularization=regularization,kernel_activation = 'relu')  # , regularization=parser_result.regularization)

	def get_name(self):
		return "Auto Encoder Super Resolution"


def get_model_interface() -> ModelBase:
	return AESuperResolutionModel()


def create_dscr_auto_encoder_model(input_shape: tuple, output_shape: tuple, use_resnet: bool, regularization: float, kernel_activation: str):

	use_batch_norm : bool = True

	init = tf.keras.initializers.GlorotUniform()

	input = layers.Input(shape=input_shape)
	number_layers : int = 2
	offset_degre : int = 6

	x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
					  kernel_initializer=init)(input)
	if use_batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = create_activation(kernel_activation)(x)

	lastSumLayer = x

	for i in range(0, number_layers):
		filter_size = 2 ** (i + 6)
		filter_size = min(filter_size, 1024)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=2, kernel_initializer=init)(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		#x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=2, kernel_initializer=init)(x)
		#if use_batch_norm:
		#	x = layers.BatchNormalization(dtype='float32')(x)
		#x = create_activation(kernel_activation)(x)

		AttachLayer = x
		if use_resnet:
			if lastSumLayer is not None:
				lastSumLayer = layers.Conv2D(filters=filter_size, kernel_size=(1, 1), kernel_initializer=init,
											 strides=(2, 2))(lastSumLayer)
				encoder_last_conv2 = lastSumLayer
				x = layers.add([AttachLayer, lastSumLayer])

			lastSumLayer = x
			x = create_activation(kernel_activation)(x)

	connect_conv_shape = x.shape

	x = layers.Flatten(name="latentspace")(x)
	x = layers.ActivityRegularization(l1=10 ** -regularization)(x)

	x = layers.Reshape(target_shape=(
		connect_conv_shape[1], connect_conv_shape[2], connect_conv_shape[3]))(x)

	lastSumLayer = x
	for i in range(0, number_layers + 1):
		filter_size = 2 ** (6 + number_layers - i)
		filter_size = min(filter_size, 1024)

		x = layers.UpSampling2D(size=(2, 2))(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if use_batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		AttachLayer = x
		if use_resnet:
			if lastSumLayer is not None:
				lastSumLayer = layers.Conv2DTranspose(filters=filter_size, kernel_size=(
					1, 1), kernel_initializer=init, strides=(2, 2))(lastSumLayer)
				x = layers.add([AttachLayer, lastSumLayer])
			lastSumLayer = x
			x = create_activation(kernel_activation)(x)

	x = layers.Conv2DTranspose(filters=3, kernel_size=(9, 9), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)

	# Confirm the output shape.
	assert x.shape[1:] == output_shape

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder
