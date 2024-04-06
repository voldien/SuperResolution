import math
from ast import Dict

from matplotlib import pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from PIL import Image


def convert_nontensor_color_space(image_data, color_space: str):
	"""_summary_

	Args:
		image_data (_type_): _description_
		color_space (str): _description_

	Returns:
		_type_: [0,1]
	"""
	if color_space == 'lab':
		if isinstance(image_data, list):
			# Convert [-1,1] -> [-128,128] -> [0,1]
			return np.asarray([lab2rgb(image * 128.0) for image in image_data]).astype(dtype='float32')
		else:
			return lab2rgb(image_data * 128.0)
	# Convert [-1,1] -> [0,1]
	elif color_space == 'rgb':
		return (image_data + 1.0) * 0.5
	else:
		assert 0


# TODO: add
# @tf.function
# def setup_color_encoding(img, color_space: str):
# 	#TODO relocate to its own method.
# 	@tf.function
# 	def preprocess_rgb2lab(tensorData):

# 		image = tf.cast(tensorData, float_precision)

# 		return tf.cast(tfio.experimental.color.rgb_to_lab(image), float_precision)

# 	# Convert color space encoding and normalize values.
# 	if color_space == 'lab':
# 		# Convert to LAB color and Transform [0, 255] -> [-128,128] -> [-1, 1]
# 		return preprocess_rgb2lab(normalization_layer(img)) * (1.0 / 128.0)
# 	elif color_space == 'rgb':
# 		# Normalize and Transform [0, 255] -> [0,1] -> [-1, 1]
# 		return (normalization_layer(img) * 2.0) - 1.0
# 	assert 0

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

	# iterate through each image tile.
	for decoder_image in decoder_images:
		# Clip to valid color value and convert to uint8.
		decoder_image = decoder_image.clip(0.0, 1.0)
		decoder_image_u8 = np.uint8((decoder_image * 255).round())

		# Convert numpy to Image.
		compressed_crop_im = Image.fromarray(decoder_image_u8, "RGB")

		packed_cropped_result.append(compressed_crop_im)

	return packed_cropped_result


def upscale_composite_image(upscale_model : tf.keras.Model, input_im: Image, batch_size: int, color_space: str):
	image_input_shape: tuple = upscale_model.input_shape[1:]
	image_output_shape: tuple = upscale_model.output_shape[1:]

	#
	input_width, input_height, input_channels = image_input_shape
	output_width, output_height, output_channels = image_output_shape

	#
	width_scale: float = float(output_width) / float(input_width)
	height_scale: float = float(output_height) / float(input_height)

	# Convert to RGB Color Space.
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

		#
		crop_batch: list = []
		for crop in cropped_batch:
			cropped_sub_input_image = input_im.crop(crop)
			crop_batch.append(np.array(cropped_sub_input_image))

		#
		normalized_subimage_color = (np.array(crop_batch) * (1.0 / 255.0)).astype(
			dtype='float32')

		# COnmvert RGB to the color space used by the model.
		if color_space == 'lab':
			# [0,1] -> [-128,128] ->  [-1,1]
			cropped_sub_input_image = rgb2lab(normalized_subimage_color) * (1.0 / 128.0)
		elif color_space == 'rgb':
			# [0,1] -> [-1,1]
			cropped_sub_input_image = (normalized_subimage_color * 2) - 1
		else:
			assert 0

		# Upscale.
		upscale_raw_result = upscale_image_func(upscale_model, cropped_sub_input_image,
												color_space=color_space)

		#
		for index, (crop, upscale) in enumerate(zip(cropped_batch, upscale_raw_result)):
			output_left = int(crop[0] * width_scale)
			output_top = int(crop[1] * width_scale)
			output_right = int(crop[2] * width_scale)
			output_bottom = int(crop[3] * width_scale)

			upscale_image.paste(upscale, (output_left, output_top, output_right, output_bottom))

	# Offload final crop and save to seperate thread.
	final_cropped_size = (0, 0, upscale_new_size[0], upscale_new_size[1])
	return final_cropped_size, upscale_image


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
