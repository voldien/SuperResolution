import os
from random import randrange

import keras.callbacks
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import layers

from util.image import generate_grid_image

from util.image import showResult


def generate_latentspace(generator_model, disc_model_features, latent_spaces, dataset):
	generated_result = generator_model.predict(latent_spaces, batch_size=16, verbose=1)

	disc_model_features = get_last_multidim_model(disc_model_features)

	generated_predicted_features = disc_model_features.predict(generated_result, batch_size=16, verbose=1)
	generated_features = np.asarray(generated_predicted_features).astype('float32')  # .reshape(-1, 1)

	real_predicted_features = disc_model_features.predict(dataset, batch_size=16,
														  verbose=1)
	real_predicted_features = np.asarray(real_predicted_features).astype('float32')  # .reshape(-1, 1)

	fig = plt.figure(figsize=(10, 10), dpi=300)
	# generated_result = generator_model.predict(latent_spaces, batch_size=16, verbose=0)
	if len(generated_features[0]) > 1:
		tsne = TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto')
		#
		generated_tsne = tsne.fit_transform(generated_features)
		real_tsne = tsne.fit_transform(real_predicted_features)

		# Plot Result

		ax = plt.subplot(1, 1, 0 + 1)
		ax.title.set_text(str.format('Latent Space {0}', len(latent_spaces)))
		plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], color='blue')
		plt.scatter(real_tsne[:, 0], real_tsne[:, 1], color='red')
		plt.legend()
		plt.colorbar()
		return fig
	return fig


def get_last_multidim_model(model):
	last_multi_layer = None

	for layer in reversed(model.layers):
		nr_elements = 1
		# Skip batch size.
		for i in range(1, len(layer.output_shape)):
			nr_elements *= layer.output_shape[i]
		if nr_elements > 1:
			last_multi_layer = layer
			break

	feature_model = tf.keras.models.Model(inputs=model.input,
										  outputs=layers.Flatten()(
											  last_multi_layer.output))
	return feature_model


def plotCostHistory(results, loss_label="", val_label="", title="", x_label="", y_label=""):
	fig = plt.figure(figsize=(10, 10), dpi=300)
	for i, result in enumerate(results):
		# ax = plt.subplot(1, len(results), i + 1)
		plt.plot(result[1], label=result[0])
		plt.ylabel(ylabel=y_label)
		plt.xlabel(xlabel=x_label)
		plt.legend(loc="upper left")
	return fig


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

class SaveHistory(tf.keras.callbacks.Callback):
	def __init__(self, output_path, **kwargs):
		super(tf.keras.callbacks.Callback, self).__init__(**kwargs)
		self.output_path = output_path

	def on_epoch_begin(self, epoch, logs=None):
		pass
		#historyfig = plotCostHistory([("generator", each_batch_generator_loss_result),
		#							  ("discriminator", each_batch_discriminator_loss_result)], x_label="Loss",
		#							 y_label="Batch Iteration")
		#historyfig.savefig(os.path.join(self.output_path, "SuperResolutionCost.png"))
		#plt.close()
