from ast import Dict
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


def compute_PSNR(orignal, data):
	pass

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

def plotFitHistory(result_collection : Dict, loss_label="", val_label="", title="", x_label="", y_label=""):
	fig = plt.figure(figsize=(10, 10), dpi=300)

	for i, result_key in enumerate(result_collection.keys()):
		dataplot = result_collection[result_key]
		plt.plot(dataplot, label=result_key)
		plt.ylabel(ylabel=y_label)
		plt.xlabel(xlabel=x_label)
		plt.legend(loc="upper left")
	return fig

