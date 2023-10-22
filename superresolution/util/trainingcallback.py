import os
from random import randrange

import keras.callbacks
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from util.util import plotFitHistory
import tensorflow as tf
from tensorflow.keras import layers
from util.image import generate_grid_image


from math import log10, sqrt 
from util.image import showResult


def compute_PSNR(orignal, data):
	mse = np.mean((orignal - data) ** 2) 
	if(mse == 0):  # MSE is zero means no noise is present in the signal . 
				# Therefore PSNR have no importance. 
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse)) 
	return psnr

class PBNRImageResultCallBack(tf.keras.callbacks.Callback):
	def __init__(self, dir_path, train_data_subset, color_space, **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)		

		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		self.trainSet = train_data_subset.with_options(options)

		self.dir_path = dir_path
		self.current_epoch = 0

		self.fig = plt.figure(figsize=(10, 10), dpi=300)
		self.pbnr_history = []

	def on_epoch_begin(self, epoch, logs=None):
		self.current_epoch = epoch

		batch_iter = iter( self.trainSet)
		image_batch, _ = next(batch_iter)

		output = self.model.predict(image_batch, verbose=0)

		self.pbnr_history.append(compute_PSNR(image_batch,output))

		# ax = plt.subplot(1, len(results), i + 1)
		plt.plot(self.pbnr_history)
		plt.ylabel(ylabel="y_label")
		plt.xlabel(xlabel="x_label")
		plt.legend(loc="upper left")

		self.fig.savefig(os.path.join(self.dir_path, "GANCost.png"))

class SaveExampleResultImageCallBack(tf.keras.callbacks.Callback):

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



#def compare_images(target, ref):
#    scores = []
#    scores.append(psnr(target, ref))
#    scores.append(mse(target, ref))
#    scores.append(ssim(target, ref, multichannel=True))
#    return scores
#TODO add support with PNSS, to extract how good it is in contrast to the original data.
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
	def __init__(self, filepath : str):
		super().__init__()
		self.fig_savepath = filepath

	def on_train_begin(self, logs):
		super().on_train_begin(logs=logs)

	def on_epoch_end(self, epoch, logs):
		super().on_epoch_end(epoch=epoch, logs=logs)

		fig = plotFitHistory(self.history, x_label="Epoch", y_label="value")
		fig.savefig(self.fig_savepath)