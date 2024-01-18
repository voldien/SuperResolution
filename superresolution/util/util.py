from ast import Dict

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage.color import lab2rgb
from sklearn.manifold import TSNE
from tensorflow.keras import layers


def convert_nontensor_color_space(image_data, color_space: str):
	"""_summary_

	Args:
		image_data (_type_): _description_
		color_space (str): _description_

	Returns:
		_type_: _description_
	"""
	if color_space == 'lab':
		if isinstance(image_data, list):
			# Convert [-1,1] -> [-128,128] -> [0,1]
			return np.asarray([lab2rgb(image * 128.0) for image in image_data]).astype(dtype='float32')
		else:
			return lab2rgb(image_data * 128.0)
	elif color_space == 'rgb':
		return (image_data + 1.0) * 0.5
	else:
		assert 0

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


def plotTrainingHistory(result_collection: Dict, loss_label="", val_label="", title="", x_label="", y_label=""):
	fig = plt.figure(figsize=(10, 10), dpi=300)

	for i, result_key in enumerate(result_collection.keys()):
		dataplot = result_collection[result_key]
		plt.plot(dataplot, label=result_key)
		plt.ylabel(ylabel=y_label)
		plt.xlabel(xlabel=x_label)
		plt.legend(loc="upper left")
	return fig
