# !/usr/bin/env python3
import argparse
import logging
import math
import os
import sys
import traceback
from logging import Logger
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.color import rgb2lab

from core.common import setup_tensorflow_strategy
from util.util import upscale_image_func

sr_logger: Logger = logging.getLogger('SuperResolution Upscale Program')
#
console_handler = logging.StreamHandler()
log_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(log_format))

sr_logger.addHandler(console_handler)


def upscale_logic(input_im: Image, image_input_size: tuple, image_scale: tuple, batch_size, color_space, upscale_model):
	# Open File and Convert to RGB Color Space.
	input_im: Image = input_im.convert('RGB')
	width_scale, height_scale = image_scale
	input_width, input_height, _ = image_input_size

	#
	upscale_new_size: tuple = (
		int(input_im.size[0] * width_scale), int(input_im.size[1] * height_scale))
	sr_logger.info("Upscale Size " + str(upscale_new_size))

	# New Upscale Size.
	upscale_image = Image.new("RGB", upscale_new_size, (0, 0, 0))

	#
	nr_width_block: int = math.ceil(
		float(input_im.width) / float(input_width))
	nr_height_block: int = math.ceil(
		float(input_im.height) / float(input_height))

	sr_logger.debug(str.format(
		"Number of tiles: {0}:{1}", nr_width_block, nr_height_block))

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
	nr_cropped_batches: int = int(
		math.ceil(len(image_crop_list) / batch_size))

	#
	for nth_batch in range(0, nr_cropped_batches):
		cropped_batch = image_crop_list[nth_batch *
		                                batch_size:(nth_batch + 1) * batch_size]

		crop_batch = []
		for crop in cropped_batch:
			cropped_sub_input_image = input_im.crop(crop)
			crop_batch.append(np.array(cropped_sub_input_image))

		normalized_subimage_color = (np.array(crop_batch) * (1.0 / 255.0)).astype(
			dtype='float32')

		# TODO fix color space conversation, use function.
		if color_space == 'lab':
			cropped_sub_input_image = rgb2lab(
				normalized_subimage_color) * (1.0 / 128.0)
		elif color_space == 'rgb':
			cropped_sub_input_image = (
				                          normalized_subimage_color * 2) - 1

		# Upscale.
		upscale_raw_result = upscale_image_func(upscale_model, cropped_sub_input_image,
		                                        color_space=color_space)

		#
		for index, (crop, upscale) in enumerate(zip(cropped_batch, upscale_raw_result)):
			output_left = int(crop[0] * width_scale)
			output_top = int(crop[1] * width_scale)
			output_right = int(crop[2] * width_scale)
			output_bottom = int(crop[3] * width_scale)

			upscale_image.paste(
				upscale, (output_left, output_top, output_right, output_bottom))

	# Offload final crop and save to separate thread.
	final_cropped_size = (
		0, 0, upscale_new_size[0], upscale_new_size[1])
	return upscale_image, final_cropped_size


def save_result_file(argument):
	upscale_image, new_cropped_size, full_output_path = argument

	# Crop image final size image.
	upscale_image = upscale_image.crop(new_cropped_size)
	# Save image
	try:
		sr_logger.info("Saving Image {0}".format(full_output_path))
		upscale_image.save(full_output_path)
	except Exception as excep:
		sr_logger.error(excep)


def create_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description='UpScale')

	#
	parser.add_argument('--save-output', dest='save_path',
	                    default=None, type=str, help='')

	#
	parser.add_argument('--input-file', type=str,  # action='append', TODO:add later
	                    default=None, dest='input_files')

	#
	parser.add_argument('--batch-size', type=int, default=16, dest='batch_size',
	                    help='number images processed at the same time.')

	#
	parser.add_argument('--threads', type=int, default=1, dest='task_threads',
	                    help='number images processed at the same time.')

	#
	parser.add_argument('--model', dest='model_filepath',
	                    default=None, help='')

	#
	parser.add_argument('--model-weight', dest='model_weight_path', type=str,
	                    help='Select Model Weight Path', default=None)

	#
	parser.add_argument('--device', action='append', default=None, required=False,
	                    dest='devices', help='Select the device explicitly that will be used.')

	parser.add_argument('--cpu', action='store_true',
	                    default=False,
	                    dest='use_explicit_cpu', help='Explicit use the CPU as the compute device.')

	#
	parser.add_argument('--verbosity', type=int, dest='accumulate',
	                    default=1,
	                    help='')

	#
	parser.add_argument('--debug', action='store_true', dest='debug', help='')

	#
	parser.add_argument('--color-space', type=str, default="rgb", dest='color_space', choices=['rgb', 'lab'],
	                    help='Select Color Space Image wisll be decode from the output model data.')

	#
	parser.add_argument('--color-channels', type=int, default=3, dest='color_channels', choices=[1, 3, 4],
	                    help='Select Number of channels in the color space. GrayScale, RGB and RGBA.')
	return parser


def super_resolution_upscale(argv):
	parser = create_parser()

	args = parser.parse_args(args=argv)

	sr_logger.setLevel(logging.INFO)
	if args.debug:
		sr_logger.setLevel(logging.DEBUG)

	# Allow to use multiple GPU
	strategy = setup_tensorflow_strategy(args=args)
	sr_logger.info('Number of devices: {0}'.format(
		strategy.num_replicas_in_sync))
	with strategy.scope():

		# TODO: fix output.
		output_path: str = args.save_path
		if not os.path.exists(output_path):
			os.mkdir(output_path)

		# TODO improved extraction of filepaths.
		input_filepaths: str = args.input_files

		if os.path.isdir(input_filepaths):
			sr_logger.info("Directory Path: " + str(input_filepaths))
			all_files = os.listdir(input_filepaths)
			base_bath = input_filepaths
			input_filepaths: list = [os.path.join(
				base_bath, path) for path in all_files]
		else:  # Convert to list
			sr_logger.info("File Path: " + str(input_filepaths))
			input_filepaths: list = [input_filepaths]

		batch_size: int = args.batch_size * strategy.num_replicas_in_sync

		sr_logger.info("Number of files {0}".format(len(input_filepaths)))

		upscale_model = tf.keras.models.load_model(
			filepath=args.model_filepath, compile=False)

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
		sr_logger.info("Upscale X" + str(width_scale) +
		               " UpscaleY " + str(height_scale))

		# Create a pool of task scheduler.
		pool = Pool(processes=16)

		for input_file_path in input_filepaths:

			if not os.path.isfile(input_file_path):
				continue
			sr_logger.info("Starting Image {0}".format(input_file_path))

			#
			base_filepath: str = os.path.basename(input_file_path)
			full_output_path: str = os.path.join(
				output_path, base_filepath)

			# Open File and Convert to RGB Color Space.
			input_im: Image = Image.open(input_file_path)

			upscale_image, final_cropped_size = upscale_logic(input_im=input_im, image_input_size=image_input_shape,
			                                                  image_scale=(
				                                                  width_scale, height_scale), batch_size=batch_size,
			                                                  color_space=color_space, upscale_model=upscale_model)

			sr_logger.debug(str.format("Saving {0}", full_output_path))
			pool.apply_async(save_result_file, [
				(upscale_image, final_cropped_size, full_output_path)])
		# Close and wait intill all tasks has been finished.
		pool.close()
		pool.join()


# If running the script as main executable
if __name__ == '__main__':
	try:
		super_resolution_upscale(sys.argv[1:])
	except Exception as ex:
		sr_logger.error(ex)
		print(ex)
		traceback.print_exc()
