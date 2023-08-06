import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_simple_model(input_shape, output_shape):
	batch_norm = True

	init = tf.keras.initializers.Orthogonal()

	input = layers.Input(shape=input_shape)
	number_layers = 3

	x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
					  kernel_initializer=init)(input)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)

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

	x = tf.nn.depth_to_space(x, 2)

	x = layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)
	#x = layers.ActivityRegularization(l1=0.0001)(x)

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder

def create_simple_model3(input_shape, output_shape):
	batch_norm = False

	init = tf.keras.initializers.Orthogonal()

	input = layers.Input(shape=input_shape)
	number_layers = 3

	x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
					  kernel_initializer=init)(input)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)

	for i in range(0, number_layers):
		filter_size = 2 ** (i + 5)
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

	#x = tf.nn.depth_to_space(x, 2)

	#x = layers.Conv2DTranspose(filters=filter_size, kernel_size=(3, 3), strides=(2, 2), use_bias=False,
	#						   padding='same',
	#						   kernel_initializer=init)(x)

	#for i in range(0, 1):
	#	filter_size = 2 ** (7)
	#	filter_size = min(filter_size, 1024)
#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)
#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)
#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)

	x = layers.Conv2D(filters=3, kernel_size=(9, 9), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)
	#x = layers.ActivityRegularization(l1=0.0001)(x)

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder

def create_simple_model2(input_shape, output_shape):
	batch_norm = False

	init = tf.keras.initializers.TruncatedNormal(stddev=0.02)

	input = layers.Input(shape=input_shape)
	number_layers = 3

	x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
					  kernel_initializer=init)(input)
	if batch_norm:
		x = layers.BatchNormalization(dtype='float32')(x)
	x = layers.ReLU(dtype='float32')(x)

	for i in range(0, number_layers):
		filter_size = 2 ** (i + 5)
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

	# AttachLayer = x
	# if use_resnet:
	#	if lastSumLayer is not None:
	#		lastSumLayer = layers.Conv2D(filters=filter_size, kernel_size=(1, 1), kernel_initializer=kernel_init,
	#									 strides=(2, 2))(lastSumLayer)
	#		encoder_last_conv2 = lastSumLayer
	#		x = layers.add([AttachLayer, lastSumLayer])
	#
	#	x = create_activation(kernel_activation)(x)
	# lastSumLayer = x

	x = layers.Conv2DTranspose(filters=filter_size, kernel_size=(3, 3), strides=(2, 2), use_bias=False,
							   padding='same',
							   kernel_initializer=init)(x)

	#for i in range(0, 1):
	#	filter_size = 2 ** (7)
	#	filter_size = min(filter_size, 1024)
#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)
#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)
#
	#	x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer=init)(x)
	#	if batch_norm:
	#		x = layers.BatchNormalization(dtype='float32')(x)
	#	x = layers.ReLU(dtype='float32')(x)

	x = layers.Conv2DTranspose(filters=3, kernel_size=(9, 9), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)
	x = layers.ActivityRegularization(l1=0.0001)(x)

	conv_autoencoder = keras.Model(inputs=input, outputs=x)
	return conv_autoencoder
