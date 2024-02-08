import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image as Img
import numpy as np
from util.image import show_expect_predicted_result
from util.util import plotTrainingHistory, upscale_composite_image, convert_nontensor_color_space


def compute_normalized_PSNR(orignal, data):
	return tf.image.psnr((orignal + 1) * 5, (data + 1) * 5, max_val=10.0)


class SaveExampleResultImageCallBack(tf.keras.callbacks.Callback):

	def __init__(self, dir_path, train_data_subset, color_space: str, nth_batch_sample: int = 512, grid_size: int = 6,
				 **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)

		self.current_epoch = 0

		self.trainSet = train_data_subset
		self.dir_path = dir_path

		self.nth_batch_sample = nth_batch_sample
		self.color_space = color_space
		self.grid_size = grid_size

		if not os.path.exists(self.dir_path):
			os.mkdir(self.dir_path)

	def on_epoch_begin(self, epoch, logs=None):
		self.current_epoch = epoch

	def on_epoch_end(self, epoch, logs=None):
		fig = show_expect_predicted_result(model=self.model, image_batch_dataset=self.trainSet,
										   color_space=self.color_space, nr_col=self.grid_size)
		fig.savefig(os.path.join(self.dir_path, "SuperResolution{0}.png".format(epoch)))
		fig.clf()
		plt.close(fig)

	def on_train_batch_end(self, batch, logs=None):
		if batch % self.nth_batch_sample == 0:
			fig = show_expect_predicted_result(model=self.model, image_batch_dataset=self.trainSet,
											   color_space=self.color_space, nr_col=self.grid_size)
			fig.savefig(os.path.join(self.dir_path, "SuperResolution_{0}_{1}.png".format(self.current_epoch, batch)))
			fig.clf()
			plt.close(fig)


class CompositeImageResultCallBack(tf.keras.callbacks.Callback):

	def __init__(self, dir_path: str, name: str, train_data_subset, color_space: str, num_images: int = 1, **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)

		self.trainSet = train_data_subset

		self.composite_name = name
		self.dir_path = dir_path
		self.current_epoch = 0
		self.color_space = color_space
		self.num_images = num_images
		if not os.path.exists(self.dir_path):
			os.mkdir(self.dir_path)

	def on_epoch_begin(self, epoch, logs=None):
		self.current_epoch = epoch

		# TODO: relocate to its own function
		batch_iter = iter(self.trainSet)
		_, expected_image_batch = batch_iter.next()

		for i in range(0, self.num_images):


			# Convert image to normalized [0,1]
			data_image = np.asarray(
				convert_nontensor_color_space(expected_image_batch[i], color_space=self.color_space)).astype(
				dtype='float32')			#TODO: add iterate fix when exceed the size.

			# Clip and convert it to [0,255], for RGB.
			data_image = data_image.clip(0.0, 1.0)
			decoder_image_u8 = np.uint8((data_image * 255).round())

			input_rgb_image = Img.fromarray(decoder_image_u8, mode='RGB')

			final_cropped_size, upscale_image = upscale_composite_image(upscale_model=self.model, input_im=input_rgb_image,
																		batch_size=16, color_space=self.color_space)

			full_output_path = os.path.join(self.dir_path,
											"SuperResolution_Composite_{0}_{1}_{2}.png".format(self.composite_name,
																						self.current_epoch, i))
			# Crop image final size image.
			upscale_image = upscale_image.crop(final_cropped_size)
			# Save image
			upscale_image.save(full_output_path)


class EvoluteSuperResolutionPerformance(tf.keras.callbacks.Callback):

	def __init__(self, dir_path, train_data_subset, color_space, **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)

		self.trainSet = train_data_subset

		self.dir_path = dir_path
		self.current_epoch = 0
		self.nth_batch_sample = 512
		self.color_space = color_space
		if not os.path.exists(self.dir_path):
			os.mkdir(self.dir_path)

	def on_epoch_begin(self, epoch, logs=None):
		self.current_epoch = epoch

	def on_epoch_end(self, epoch, logs=None):
		fig = show_expect_predicted_result(model=self.model, image_batch_dataset=self.trainSet,
										   color_space=self.color_space)
		fig.savefig(os.path.join(self.dir_path, "SuperResolution{0}.png".format(epoch)))
		fig.clf()
		plt.close(fig)

	def on_train_batch_end(self, batch, logs=None):
		if batch % self.nth_batch_sample == 0:
			fig = show_expect_predicted_result(model=self.model, image_batch_dataset=self.trainSet,
											   color_space=self.color_space)
			fig.savefig(os.path.join(self.dir_path, "SuperResolution_{0}_{1}.png".format(self.current_epoch, batch)))
			fig.clf()
			plt.close(fig)


class GraphHistory(tf.keras.callbacks.History):
	def __init__(self, filepath: str, nthBatch: int = 8, **kwargs):
		super().__init__(**kwargs)
		self.fig_savepath = filepath
		self.nthBatch = nthBatch
		self.batch_history = {}

	def on_train_begin(self, logs=None):
		super().on_train_begin(logs=logs)

	def on_train_batch_end(self, batch, logs=None):
		super().on_train_batch_end(batch=batch, logs=logs)
		if batch % self.nthBatch == 0:
			for k, v in logs.items():
				self.batch_history.setdefault(k, []).append(v)
			# Append learning rate. #TODO fix how to extract learning rate
			learning_rate = 0.0
			if isinstance(self.model.optimizer.lr, tf.keras.optimizers.schedules.ExponentialDecay):
				learning_rate = 0.0  # self.model.optimizer.lr()
			self.batch_history.setdefault("learning-rate", []).append(learning_rate)

	def on_epoch_end(self, epoch, logs=None):
		super().on_epoch_end(epoch=epoch, logs=logs)

		# Plot detailed
		fig = plotTrainingHistory(self.batch_history, x_label="Batches", y_label="value")
		fig.savefig(self.fig_savepath)
		fig.clf()
		plt.close(fig)
