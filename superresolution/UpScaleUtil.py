# !/usr/bin/env python3
import argparse
import logging
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.color import rgb2lab

from util.util import convert_nontensor_color_space


# def pixel_shuffle(scale):
#	return lambda x: tf.nn.depth_to_space(x, scale)


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


def save_result_file(argument):
	upscale_image, new_cropped_size, full_output_path = argument

	# Crop image final size image.
	upscale_image = upscale_image.crop(new_cropped_size)
	# Save image
	logger.info("Saving Image {0}".format(full_output_path))
	upscale_image.save(full_output_path)


def super_resolution_upscale(argv):
	parser = argparse.ArgumentParser(
		description='UpScale')

	#
	parser.add_argument('--save-output', dest='save_path', default=None, help='')

	#
	parser.add_argument('--batch-size', type=int, default=16, dest='batch_size',
						help='number images processed at the same time.')

	#
	parser.add_argument('--threads', type=int, default=1, dest='task_threads',
						help='number images processed at the same time.')

	#
	parser.add_argument('--model', dest='model_filepath', default=None, help='')

	#
	parser.add_argument('--input-file', action='store', dest='input_files')

	#
	parser.add_argument('--model-weight', action='store', dest='model_weight_path', type=str,
						help='Select Model Weight Path', default=None)

	#
	parser.add_argument('--device', type=str, dest='', default=None, help='Select Device')

	#
	parser.add_argument('--verbosity', type=int, dest='accumulate',
						default=1,
						help='')

	#
	parser.add_argument('--color-space', type=str, default="rgb", dest='color_space', choices=['rgb', 'lab'],
						help='Select Color Space Image wisll be decode from the output model data.')

	#
	parser.add_argument('--color-channels', type=int, default=3, dest='color_channels', choices=[1, 3, 4],
						help='Select Number of channels in the color space. GrayScale, RGB and RGBA.')

	args = parser.parse_args(args=argv)

	global logger
	logger = logging.getLogger('SuperResolution Upscale')
	logger.setLevel(logging.INFO)

	with tf.device('/device:GPU:0'):

		# Open files, convert to rgb.
		output_path: str = args.save_path
		if os.path.isdir(output_path):
			pass

		# TODO improved extraction of filepaths.
		input_filepaths: str = args.input_files
		logging.info(input_filepaths)
		if os.path.isdir(input_filepaths):
			all_files = os.listdir(input_filepaths)
			base_bath = input_filepaths
			input_filepaths: list = [os.path.join(base_bath, path) for path in all_files]
		else:  # Convert to list
			input_filepaths: list = [input_filepaths]

		batch_size = args.batch_size

		logger.info("Number of files {0}".format(len(input_filepaths)))

		upscale_model = tf.keras.models.load_model(filepath=args.model_filepath, compile=False)

		# Optionally, load specific weight.
		if args.model_weight_path:
			upscale_model.load_weights(filepath=args.model_weight_path)

		color_space: str = args.color_space

		upscale_model.summary()

		#
		image_input_shape: tuple = upscale_model.input_shape[1:]
		image_output_shape: tuple = upscale_model.output_shape[1:]

		#
		input_width, input_height, input_channels = image_input_shape
		output_width, output_height, output_channels = image_output_shape

		#
		width_scale: float = float(output_width) / float(input_width)
		height_scale: float = float(output_height) / float(input_height)
		logger.info("Upscale X" + str(width_scale) + " UpscaleY " + str(height_scale))

		# Create a pool of task scheduler.
		with Pool(processes=10) as p:

			# TODO add batch, if possible.

			for file_path in input_filepaths:
				if not os.path.isfile(file_path):
					continue
				logger.info("Starting Image {0}".format(file_path))

				#
				base_filepath: str = os.path.basename(file_path)
				full_output_path: str = os.path.join(output_path, base_filepath)

				# Open File and Convert to RGB Color Space.
				input_im: Image = Image.open(file_path)
				input_im: Image = input_im.convert('RGB')

				#
				upscale_new_size: tuple = (int(input_im.size[0] * width_scale), int(input_im.size[1] * height_scale))
				logger.info("Upscale Size " + str(upscale_new_size))

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
				p.map_async(save_result_file, [(upscale_image, final_cropped_size, full_output_path)])


# If running the script as main executable
if __name__ == '__main__':
	try:
		super_resolution_upscale(sys.argv[1:])
	except:
		pass
