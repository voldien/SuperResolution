import argparse

import tensorflow as tf
from core import ModelBase
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras import ops
import math


class GANSuperResolutionModel(ModelBase):
	def __init__(self):
		self.parser = argparse.ArgumentParser(add_help=False)
		self.possible_upscale = [2, 4]
		#
		self.parser.add_argument('--regularization', dest='regularization',
								 type=float,
								 default=0.00001,
								 required=False,
								 help='Set the L1 Regularization applied.')

		#
		self.parser.add_argument('--use-resnet', type=bool, default=False, dest='use_resnet',
								 help='Set the number of passes that the training set will be trained against.')

	def load_argument(self) -> argparse.ArgumentParser:
		#
		return self.parser

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		scale_factor: int = int(output_shape[0] / input_shape[0])
		scale_factor: int = int(output_shape[1] / input_shape[1])

		if scale_factor not in self.possible_upscale and scale_factor not in self.possible_upscale:
			raise ValueError("Invalid upscale")

		regularization: float = kwargs.get("regularization", 0.000001)  #
		upscale_mode: int = scale_factor  #

		#
		return create_gan_model(input_shape=input_shape,
								output_shape=output_shape, upscale_count=upscale_mode,
								regularization=regularization)

	def get_name(self):
		return "SuperResolution - GAN - Generative Adversarial Network"


def get_model_interface() -> ModelBase:
	return GANSuperResolutionModel()


def residual_block(input, filters, init):

	x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
					  padding='same',
					  kernel_initializer=init)(input)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
					  padding='same',
					  kernel_initializer=init)(x)
	x = layers.BatchNormalization()(x)

	AttachLayer = x

	x = layers.add([AttachLayer, input])
	return x


def upsample(x, scale: int, num_filters: int):
	def upsample_1(input_layer, factor, **kwargs):
		"""Sub-pixel convolution."""
		x_ = layers.Conv2D(filters=num_filters * (factor ** 2), kernel_size=(3, 3), padding='same', use_bias=True, **kwargs)(
			input_layer)
		return tf.nn.depth_to_space(x_, 2)

	if scale == 2:
		x = upsample_1(x, 2, name='conv2d_1_scale_2')
	elif scale == 3:
		x = upsample_1(x, 3, name='conv2d_1_scale_3')
	elif scale == 4:
		x = upsample_1(x, 2, name='conv2d_1_scale_2')
		x = upsample_1(x, 2, name='conv2d_2_scale_2')

	return x


def make_generator_model(input_shape, output_shape, upscale_num: int = 2,  num_res_blocks: int = 8, **kwargs):
	use_upscale = False
	use_filter_up = False
	use_residual_block = True

	output_width, output_height, output_channels = output_shape
	init = 'glorot_uniform'

	input = layers.Input(shape=input_shape)

	x = layers.Conv2D(filters=64, kernel_size=(9, 9), strides=(1, 1), use_bias=True,
					  padding='same',
					  kernel_initializer=init)(input)
	x = layers.ReLU()(x)

	if use_residual_block:
		firstLayer = x
		for _ in range(0, num_res_blocks):
			x = residual_block(filters=64, init=init, input=x)

		x = layers.Conv2D(filters=64, kernel_size=(9, 9), strides=(1, 1), use_bias=True,
						  padding='same',
						  kernel_initializer=init)(input)
		x = layers.BatchNormalization()(x)
		x = layers.add([firstLayer, x])

	x = upsample(x, upscale_num, 256)

	# output layer
	x = layers.Conv2DTranspose(filters=output_channels, kernel_size=(9, 9), strides=(
		1, 1), padding='same', kernel_initializer=init)(x)
	x = layers.Activation('tanh')(x)

	# Confirm the output shape.
	print(x.shape[1:])
	assert x.shape[1:] == output_shape

	model = tf.keras.models.Model(inputs=input, outputs=x)

	return model


def residual_block_discriminator(input, filters, init, kernel_size):
	x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), use_bias=True,
					  padding='same',
					  kernel_initializer=init)(input)
	x = layers.LeakyReLU()(x)
	x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), use_bias=True,
					  padding='same',
					  kernel_initializer=init)(x)

	AttachLayer = x
	x = layers.add([AttachLayer, input])

	x = layers.LeakyReLU()(x)

	return x


def discriminator_down_block(input, filters, strides, init, use_norm=False):
	# Downscale
	x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, use_bias=True, padding='same',
					  kernel_initializer=init)(input)
	if use_norm:
		x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU(alpha=0.2)(x)
	return x


def make_discriminator_model(input_size, regularization_l1=0.0002, use_resnet=False, use_wasserstein=False, **kwargs):
	use_norm = True

	init = 'glorot_uniform'
	n_layers = max(int(math.log2(input_size[1])-1), 0)

	input = layers.Input(shape=input_size)
	x = input

	x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
					  padding='same',
					  kernel_initializer=init)(x)
	x = layers.LeakyReLU()(x)

	# Downscale
	x = discriminator_down_block(
		input=x, filters=64, strides=(2, 2), init=init, use_norm=use_norm)

	for j in range(0, n_layers - 1):
		filter_size = 128 << j
		filter_size = min(1024, filter_size)

	x = discriminator_down_block(
		input=x, filters=128, strides=(1, 1), init=init, use_norm=use_norm)

	x = discriminator_down_block(
		input=x, filters=128, strides=(2, 2), init=init, use_norm=use_norm)

	x = discriminator_down_block(
		input=x, filters=256, strides=(1, 1), init=init, use_norm=use_norm)

	x = discriminator_down_block(
		input=x, filters=256, strides=(2, 2), init=init, use_norm=use_norm)

	x = discriminator_down_block(
		input=x, filters=512, strides=(2, 2), init=init, use_norm=use_norm)

	x = discriminator_down_block(
		input=x, filters=512, strides=(2, 2), init=init, use_norm=use_norm)

	x = discriminator_down_block(
		input=x, filters=512, strides=(2, 2), init=init, use_norm=use_norm)

 #
	x = layers.Flatten()(x)
	x = layers.Dense(1024)(x)
	x = layers.LeakyReLU(0.2)(x)
	x = layers.Dense(1)(x)

	if not use_wasserstein:
		x = layers.Activation('sigmoid')(x)
		x = layers.ActivityRegularization(l1=regularization_l1)(x)

	discriminator_model = tf.keras.models.Model(inputs=input, outputs=x)

	return discriminator_model


def create_gan_model(input_shape: tuple, output_shape: tuple, upscale_count: int, num_filters: int = 64,
					 num_res_blocks: int = 8, res_block_scaling: int = None,
					 regularization: float = 0.00001):
	"""Creates an GAN model."""

	# Create the discriminator.
	discriminator = make_discriminator_model(
		input_size=output_shape, regularization_l1=regularization)

	# Create the generator.
	generator = make_generator_model(
		input_shape=input_shape, upscale_num=upscale_count, output_shape=output_shape, num_res_blocks=num_res_blocks)

	sr_gan_model = SRGANModel(discriminator=discriminator, generator=generator,
							  input_shape=input_shape, output_shape=output_shape)

	# Build
	input_build = list(input_shape)
	input_build.insert(0, None)
	sr_gan_model.build(input_shape=input_build)

	return sr_gan_model


class SRGANModel(tf.keras.Model):
	def __init__(self, discriminator, generator, input_shape: tuple, output_shape: tuple, **kwargs):
		super(SRGANModel, self).__init__(**kwargs)
		self.discriminator = discriminator
		self.generator = generator
		self.discriminator_iterations = 1

		self.use_wgandp = False
		self.use_gradient_penalty = False

		self.gradient_loss_rate = 0.01
		self.discriminator_loss_rate = 0.0001

		self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
			from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

		self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
		self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

	@property
	def metrics(self):
		return [self.gen_loss_tracker, self.disc_loss_tracker]

	def compile(self, d_optimizer=None, g_optimizer=None, loss=None, optimizer=None, metrics=None):
		super().compile(metrics=metrics, loss=loss)

		self.generator_optimizer = optimizer  # g_optimizer
		self.discriminator_optimizer = optimizer  # d_optimizer

		self.discriminator_optimizer = tf.keras.optimizers.Adam(
			learning_rate=optimizer.learning_rate, beta_1=0.5, beta_2=0.9)

		self.loss_fn = loss
		self.gp_weight = 0.5

	def build(self, input_shape):
		super().build(input_shape=input_shape)
		self.discriminator.build(input_shape)
		self.generator.build(input_shape)
		#
		self._set_inputs(inputs=input_shape)

	def call(self, x, training=False):
		if training:
			return self.train_step(x)
		else:
			return self.generator(x, training)

	def summary(self):
		self.generator.summary()
		self.discriminator.summary()

	def get_config(self):
		config = super().get_config()

		return config

	def train_step(self, data):

		# Extract the batch size from the data chunk.
		data_batch_size = tf.shape(data[0])[0]

		# Unpack the data.
		low_res_images, upscale_image = data

		for _ in range(0, self.discriminator_iterations):

			with tf.GradientTape() as disc_tape:
				if not self.use_wgandp:

					generated_image = tf.dtypes.cast(self.generator(
						low_res_images, training=True), tf.float32)

					# discriminator's prediction for real image
					real_output = tf.dtypes.cast(self.discriminator(
						upscale_image, training=True), tf.float32)

					# discriminator's estimate for fake image
					fake_output = tf.dtypes.cast(self.discriminator(
						generated_image, training=True), tf.float32)

					# Calculate the gradient penalty
					if self.use_gradient_penalty:
						gp = tf.dtypes.cast(self.gradient_penalty(self.discriminator, data_batch_size, upscale_image, generated_image),
											tf.float32)

					disc_loss = tf.dtypes.cast(
						self.gan_discriminator_loss(self.cross_entropy, self.gradient_loss_rate, self.discriminator_loss_rate,
													real_output, fake_output), tf.float32)  # + gp * self.gp_weight  # * (

				else:
					def discriminator_loss(real_img, fake_img):
						real_loss = tf.reduce_mean(real_img)
						fake_loss = tf.reduce_mean(fake_img)
						return fake_loss - real_loss

					# Generate fake images from the latent vector
					fake_images = tf.dtypes.cast(self.generator(
						low_res_images, training=True), tf.float32)
					# Get the logits for the fake images
					fake_logits = self.discriminator(
						fake_images, training=True)
					# Get the logits for the real images
					real_logits = self.discriminator(
						upscale_image, training=True)

					# Calculate the discriminator loss using the fake and real image logits
					d_cost = tf.dtypes.cast(discriminator_loss(real_img=real_logits, fake_img=fake_logits),
											tf.float32)
					# Calculate the gradient penalty
					if self.use_gradient_penalty:
						gp = tf.dtypes.cast(self.gradient_penalty(self.discriminator, data_batch_size, upscale_image, fake_images),
											tf.float32)
						# Add the gradient penalty to the original discriminator loss
						disc_loss = d_cost + gp * self.gp_weight
					else:
						disc_loss = d_cost

				# compute gradient
				discriminator_grad = disc_tape.gradient(
					disc_loss, self.discriminator.trainable_variables)

				# update variable with gradient
				self.discriminator_optimizer.apply_gradients(
					zip(discriminator_grad, self.discriminator.trainable_variables))

			# Train Generator Model.
			with tf.GradientTape() as gen_tape:
				generated_image = tf.dtypes.cast(self.generator(
					low_res_images, training=True), tf.float32)

				fake_output = self.discriminator(
					generated_image, training=True)

				def generator_loss(fake_img):
					return -tf.reduce_mean(fake_img)

				if self.use_wgandp:
					# compute loss
					gen_loss = generator_loss(fake_output)
				else:
					# compute loss
					gen_loss = self.gan_generator_loss(
						self.gradient_loss_rate, upscale_image, generated_image, fake_output)

				# optimize generator first
				generator_grad = gen_tape.gradient(
					gen_loss, self.generator.trainable_variables)

				# optimize discriminator after generator
				self.generator_optimizer.apply_gradients(
					zip(generator_grad, self.generator.trainable_variables))

		# Monitor loss.
		self.gen_loss_tracker.update_state(gen_loss)
		self.disc_loss_tracker.update_state(disc_loss)

		return {
			"g_loss": self.gen_loss_tracker.result(),
			"d_loss": self.disc_loss_tracker.result(),
		}

	def build_graph(self, raw_shape):
		x = tf.keras.layers.Input(shape=(None, raw_shape),
								  ragged=True)

		return tf.keras.Model(inputs=[x],
							  outputs=self.call(x))

	def gan_generator_loss(self, gradient_loss_rate, real_image, generated_image, fake_output):
		# The objective is to penalize the generator whenever it produces images which the discriminator classifies as 'fake'
		content_loss = self.loss_fn(real_image, generated_image)
		adversarial_loss = 10**-3 * gradient_loss_rate * \
			self.cross_entropy(tf.ones_like(fake_output), fake_output)

		return content_loss + adversarial_loss

	def gan_discriminator_loss(self, cross_entropy, gradient_loss_rate, discriminator_loss_rate, real_output, fake_output,
							   smooth=0.01):
		# label for real image is (1-smooth)

		real_loss = cross_entropy(tf.ones_like(
			real_output) * (1 - smooth), real_output)
		#
		fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = gradient_loss_rate * fake_loss + discriminator_loss_rate * real_loss

		return total_loss

	def gradient_penalty(self, discriminator, batch_size, real_images, fake_images):
		"""	
				Calculates the gradient penalty.
				This loss is calculated on an interpolated image
				and added to the discriminator loss.
		"""
		# Get the interpolated image
		alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
		diff = fake_images - real_images
		interpolated = real_images + alpha * diff

		with tf.GradientTape() as gp_tape:
			gp_tape.watch(interpolated)
			# 1. Get the discriminator output for this interpolated image.
			pred = discriminator(interpolated, training=True)

		# 2. Calculate the gradients w.r.t to this interpolated image.
		grads = gp_tape.gradient(pred, [interpolated])[0]
		# 3. Calculate the norm of the gradients.
		norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
		gp = tf.reduce_mean((norm - 1.0) ** 2)
		return gp
