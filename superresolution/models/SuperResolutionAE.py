import argparse

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
		#
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		#
		# parser_result = self.parser.parse_known_args(sys.argv[1:])
		# Model constructor parameters.
		regularization  : float = kwargs.get("regularization", 0.00001)  #
		upscale_mode: int = kwargs.get("upscale_mode", 2)  #
		num_input_filters : int = kwargs.get("edsr_filters", 256)  #
		use_resnet = kwargs.get("use_resnet", True)  #

		#
		return create_dscr_auto_encoder_model(input_shape=input_shape,
									output_shape=output_shape, use_resnet=use_resnet, regularization=regularization)  # , regularization=parser_result.regularization)

	def get_name(self):
		return "basic super"


def get_model_interface() -> ModelBase:
	return AESuperResolutionModel()


def create_dscr_auto_encoder_model(input_shape, output_shape, use_resnet : bool, regularization:float=0.00000,kernel_activation: str='relu'):
	def create_activation(activation):
		if activation == "leaky_relu":
			return layers.LeakyReLU(alpha=0.2, dtype='float32')
		elif activation == "relu":
			return layers.ReLU(dtype='float32')
		elif activation == "sigmoid":
			return layers.Activation(activation='sigmoid', dtype='float32')
		else:
			assert "Should never be reached"

	batch_norm = False

	init = tf.keras.initializers.TruncatedNormal(stddev=0.02)

	input = layers.Input(shape=input_shape)
	number_layers = 3

	x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
					  kernel_initializer=init)(input)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = create_activation(kernel_activation)(x)

	lastSumLayer = x

	for i in range(0, number_layers ):
		filter_size = 2 ** (i + 5)
		filter_size = min(filter_size, 1024)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=2, kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)

		AttachLayer = x
		if use_resnet:
			if lastSumLayer is not None:
				lastSumLayer = layers.Conv2D(filters=filter_size, kernel_size=(1, 1), kernel_initializer=init, strides=(2, 2))(lastSumLayer)
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
		filter_size = 2 ** (7)
		filter_size = min(filter_size, 1024)
	
		x = layers.UpSampling2D(size=(2, 2))(x)
	
		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if batch_norm:
			x = layers.BatchNormalization(dtype='float32')(x)
		x = create_activation(kernel_activation)(x)
	
		x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
		if batch_norm:
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