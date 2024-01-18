import math
from ast import Dict

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from sklearn.manifold import TSNE
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.color import rgb2lab


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


def upscale_image_func(model: tf.keras.Model, image, color_space: str) -> list:
	"""_summary_

	Args:
		model (tf.keras.Model): _description_
		image (_type_): _description_
		color_space (str): _description_

	Returns:
		list: _description_
	"""
	# Perform upscale.
	result_upscale_raw = model(image, training=False)

	packed_cropped_result: list = []

	# Convert from Raw to specified ColorSpace.
	decoder_images = np.asarray(convert_nontensor_color_space(result_upscale_raw, color_space=color_space)).astype(
		dtype='float32')
	#
	for decoder_image in decoder_images:

		# Clip to valid color value and convert to uint8.
		decoder_image = decoder_image.clip(0.0, 1.0)
		decoder_image_u8 = np.uint8((decoder_image * 255).round())

		# Convert numpy to Image.
		compressed_crop_im = Image.fromarray(decoder_image_u8, "RGB")

		packed_cropped_result.append(compressed_crop_im)

	return packed_cropped_result


def upscale_composite_image(upscale_model, input_im: Image, batch_size:int, color_space:str):
	image_input_shape: tuple = upscale_model.input_shape[1:]
	image_output_shape: tuple = upscale_model.output_shape[1:]


	#
	input_width, input_height, input_channels = image_input_shape
	output_width, output_height, output_channels = image_output_shape

	#
	width_scale: float = float(output_width) / float(input_width)
	height_scale: float = float(output_height) / float(input_height)


	# Open File and Convert to RGB Color Space.
	input_im: Image = input_im.convert('RGB')

	#
	upscale_new_size: tuple = (int(input_im.size[0] * width_scale), int(input_im.size[1] * height_scale))

	#
	upscale_image = Image.new("RGB", upscale_new_size, (0, 0, 0))

	#
	nr_width_block: int = math.ceil(float(input_im.width) / float(input_width))
	nr_height_block: int = math.ceil(float(input_im.height) / float(input_height))

	# Construct all crops.
	image_crop_list: list = []
	for x in range(0, nr_width_block):
		for y in range(0, nr_height_block):
			# Compute subset view.
			left = x * input_width
			top = y * input_height
			right = (x + 1) * input_width
			bottom = (y + 1) * input_height
			image_crop_list.append((left, top, right, bottom))

	# Compute number of cropped batches.
	nr_cropped_batchs: int = int(math.ceil(len(image_crop_list) / batch_size))

	#
	for nth_batch in range(0, nr_cropped_batchs):
		cropped_batch = image_crop_list[nth_batch * batch_size:(nth_batch + 1) * batch_size]

		crop_batch = []
		for crop in cropped_batch:
			cropped_sub_input_image = input_im.crop(crop)
			crop_batch.append(np.array(cropped_sub_input_image))

		normalized_subimage_color = (np.array(crop_batch) * (1.0 / 255.0)).astype(
			dtype='float32')

		# TODO fix color space converation.
		if color_space == 'lab':
			cropped_sub_input_image = rgb2lab(normalized_subimage_color) * (1.0 / 128.0)
		elif color_space == 'rgb':
			cropped_sub_input_image = (normalized_subimage_color + 1) * 0.5
		# cropped_sub_input_image = np.expand_dims(cropped_sub_input_image, axis=0)

		# Upscale.
		upscale_raw_result = upscale_image_func(upscale_model, cropped_sub_input_image,
												color_space=color_space)

		#
		for index, (crop, upscale) in enumerate(zip(cropped_batch, upscale_raw_result)):
			# TODO fix
			output_left = int(crop[0] * width_scale)
			output_top = int(crop[1] * width_scale)
			output_right = int(crop[2] * width_scale)
			output_bottom = int(crop[3] * width_scale)

			upscale_image.paste(upscale, (output_left, output_top, output_right, output_bottom))

	# Offload final crop and save to seperate thread.
	final_cropped_size = (0, 0, upscale_new_size[0], upscale_new_size[1])
	return final_cropped_size, upscale_image

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
