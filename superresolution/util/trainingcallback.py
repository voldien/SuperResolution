import os

import tensorflow as tf
from matplotlib import pyplot as plt
from util.image import showResult
from util.util import plotTrainingHistory
import math


def compute_normalized_PSNR(orignal, data):
	return tf.psnr((orignal + 1), (data + 1), max_val=2.0)


def compute_rgb_PSNR(orignal, data):
	mse = tf.reduce_mean((orignal - data) ** 2)
	if (mse == 0.0):  # MSE is zero means no noise is present in the signal .
		# Therefore PSNR have no importance.
		return 100.0
	
	max_pixel = 255.0
	return 10 * math.log10(max_pixel) - 10 * tf.math.log(mse)

class SaveExampleResultImageCallBack(tf.keras.callbacks.Callback):

	def __init__(self, dir_path, train_data_subset, color_space, **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)

		#
		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		self.trainSet = train_data_subset.with_options(options)

		self.dir_path = dir_path
		self.current_epoch = 0
		self.nth_batch_sample = 512
		self.color_space = color_space
		if not os.path.exists(self.dir_path):
			os.mkdir(self.dir_path)

	def on_epoch_begin(self, epoch, logs=None):
		self.current_epoch = epoch

	def on_epoch_end(self, epoch, logs=None):
		fig = showResult(model=self.model, image_batch_dataset=self.trainSet, color_space=self.color_space)
		fig.savefig(os.path.join(self.dir_path, "SuperResolution{0}.png".format(epoch)))

	def on_train_batch_end(self, batch, logs=None):
		if batch % self.nth_batch_sample == 0:
			fig = showResult(model=self.model, image_batch_dataset=self.trainSet, color_space=self.color_space)
			fig.savefig(os.path.join(self.dir_path, "SuperResolution_{0}_{1}.png".format(self.current_epoch, batch)))


class EvoluteSuperResolutionPerformance(tf.keras.callbacks.Callback):

	def __init__(self, dir_path, train_data_subset, color_space, **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)

		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		self.trainSet = train_data_subset.with_options(options)

		self.dir_path = dir_path
		self.current_epoch = 0
		self.nth_batch_sample = 512
		self.color_space = color_space
		if not os.path.exists(self.dir_path):
			os.mkdir(self.dir_path)

	def on_epoch_begin(self, epoch, logs=None):
		self.current_epoch = epoch

	def on_epoch_end(self, epoch, logs=None):
		fig = showResult(model=self.model, image_batch_dataset=self.trainSet, color_space=self.color_space)
		fig.savefig(os.path.join(self.dir_path, "SuperResolution{0}.png".format(epoch)))

	def on_train_batch_end(self, batch, logs=None):
		if batch % self.nth_batch_sample == 0:
			fig = showResult(model=self.model, image_batch_dataset=self.trainSet, color_space=self.color_space)
			fig.savefig(os.path.join(self.dir_path, "SuperResolution_{0}_{1}.png".format(self.current_epoch, batch)))


class GraphHistory(tf.keras.callbacks.History):
	def __init__(self, filepath: str, **kwargs):
		super().__init__(**kwargs)
		self.fig_savepath = filepath
		self.batch_history = {}

	def on_train_begin(self, logs):
		super().on_train_begin(logs=logs)

	def on_train_batch_end(self, batch, logs=None):
		super().on_train_batch_end(batch=batch, logs=logs)
		for k, v in logs.items():
			self.batch_history.setdefault(k, []).append(v)
		# Append learning rate. #TODO fix how to extract learning rate
		learning_rate = 0.0
		if isinstance(self.model.optimizer.lr, tf.keras.optimizers.schedules.ExponentialDecay):
			learning_rate = 0.0#self.model.optimizer.lr()
		self.batch_history.setdefault("learning-rate", []).append(learning_rate)

	def on_epoch_end(self, epoch, logs):
		super().on_epoch_end(epoch=epoch, logs=logs)

		# Plot detailed
		fig = plotTrainingHistory(self.batch_history, x_label="Epoch", y_label="value")
		fig.savefig(self.fig_savepath)
