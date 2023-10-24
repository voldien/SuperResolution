import argparse
from core import ModelBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DCPostSuperResolutionModel(ModelBase):
	def __init__(self):
		pass

	def load_argument(self) -> argparse.ArgumentParser:
		"""Load in the file for extracting text."""
		parser = argparse.ArgumentParser(add_help=True, prog="", description="")

		parser.add_argument('--use-resnet', type=bool, default=False, dest='use_resnet',
							help='Set the number of passes that the training set will be trained against.')

		parser.add_argument('--override-latentspace-size', dest='generate_latentspace',
							default=False,
							help='', type=bool)
		#
		parser.add_argument('--regularization', dest='regularization',
							type=float,
							default=0.001,
							help='Set the L1 Regularization applied.')

		#
		parser.add_argument('--loss-fn', dest='loss_fn',
							default='mse',
							choices=['mses'],
							help='.', type=str)

		#
		return parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		"""Extract text from the currently loaded file."""
		return create_post_super_resolution(input_shape=input_shape, output_shape=output_shape)

	def get_name(self):
		return "post super"


def get_model_interface() -> ModelBase:
	return DCPostSuperResolutionModel()


def create_post_super_resolution(input_shape, output_shape):
	batch_norm = False

	init = tf.keras.initializers.Orthogonal()

	input = layers.Input(shape=input_shape)
	number_layers = 3

	x = layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(
		1, 1), padding='same', kernel_initializer=init)(input)
	upscale = x

	for i in range(0, number_layers):
		filter_size = 2 ** (i + 6)
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

	# x = tf.nn.depth_to_space(x, 2)

	x = layers.Conv2D(filters=3, kernel_size=(4, 4), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)

	x = layers.add([x, upscale])
	x = layers.Activation('tanh')(x)
	# x = layers.ActivityRegularization(l1=0.0001)(x)

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder
