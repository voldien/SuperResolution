# !/usr/bin/env python3
import argparse
import math
from random import randrange
import sys

from PIL import Image
import tensorflow as tf
import numpy as np
from skimage.color import lab2rgb, rgb2lab
import sys
from random import randrange
import logging
import os


def upscale_image_func(model : tf.keras.Model, image, color_space: str) -> Image:

	# 
	result_upscale_raw = model(image, training=False)

	# decoder_image = model.predict(cropped_sub_image,verbose=0) * 128.0
	decoder_image = np.asarray(lab2rgb(result_upscale_raw[0]  * 128 )).astype(dtype='float32')

	decoder_image_u8 = np.uint8(decoder_image * 255)
	compressed_crop_im = Image.fromarray(decoder_image_u8, "RGB")

	return compressed_crop_im


def super_resolution_upscale(argv):
	parser = argparse.ArgumentParser(
		description='UpScale')

	parser.add_argument('--save-output', dest='save_path', default=None, help='')

	parser.add_argument('--model', dest='model_filepath', default=None, help='')

	parser.add_argument('--input-file', action='store', dest='input_files')

	parser.add_argument('--device', type=str, dest='', default=None, help='')

	parser.add_argument('--verbosity', type=int, dest='accumulate',
						default=1,
						help='Define the save/load model path')

	parser.add_argument('--seed', type=int, default=randrange(10000000), dest='seed',
						help='Define the save/load model path')

	parser.add_argument('--color-space', type=str, default="rgb", dest='color_space', choices=['rgb', 'lab'],
						help='Select Color Space Image wisll be decode from the output model data.')
	#
	parser.add_argument('--color-channels', type=int, default=3, dest='color_channels', choices=[1, 3, 4],
						help='Select Number of channels in the color space. GrayScale, RGB and RGBA.')

	args = parser.parse_args(args=argv)

	logger = logging.getLogger('SuperResolution Upscale')
	logger.setLevel(logging.INFO)

	with tf.device('/device:CPU:0'):

		# Open files, convert to rgb.
		output_path = args.save_path
		if os.path.isdir(output_path):
			pass

		input_filepath = args.input_files
		logging.info(input_filepath)
		if os.path.isdir(input_filepath):
			input_filepath = os.listdir(input_filepath)
		else: # Convert to list
			input_filepath = [input_filepath]

		upscale_model = tf.keras.models.load_model(args.model_filepath)

		upscale_model.summary()

		image_input_shape = upscale_model.input_shape[1:]
		image_output_shape = upscale_model.output_shape[1:]

		input_width, input_height, input_channels = image_input_shape
		output_width, output_height, output_channels = image_output_shape

		for file_path in input_filepath:
			#	pass
			os.path.basename(file_path)

			width_scale = float(output_width) / float(input_width)
			height_scale = float(output_height) / float(input_height)

			logger.info("Upscale X" + str(width_scale) + " UpscaleY " + str(height_scale))

			input_im = Image.open(file_path)
			input_im = input_im.convert('RGB')

			upscale_new_size = (int(input_im.size[0] * width_scale), int(input_im.size[1] * height_scale))
			logger.info("Upscale Size " + str(upscale_new_size))

			upscale_image = Image.new("RGB", upscale_new_size, input_im.getpixel((0, 0)))

			nr_width_block = math.ceil(input_im.width / input_width)
			nr_height_block = math.ceil(input_im.height / input_height)

			# 
			for x in range(0, nr_width_block):
				for y in range(0, nr_height_block):
					# Compute subset view.
					left = x * input_width
					top = y * input_height
					right = (x + 1) * input_width
					bottom = (y + 1) * input_height

					# Create subset image.
					cropped_sub_input_image = input_im.crop((left, top, right, bottom))
					# Convert cropped subset image color space.

					normalized_subimage_color = (np.array(cropped_sub_input_image) * (1.0 / 255.0)).astype(dtype='float32')

					cropped_sub_input_image = rgb2lab(normalized_subimage_color) * (1.0 / 128.0)
					cropped_sub_input_image = np.expand_dims(cropped_sub_input_image, axis=0)

					upscale_raw_result = upscale_image_func(upscale_model, cropped_sub_input_image, color_space=args.color_space)

					# TODO fix
					output_left = int(left * width_scale)
					output_top = int(top * width_scale)
					output_right = int(right * width_scale)
					output_bottom = int(bottom * width_scale)
					upscale_image.paste(upscale_raw_result, (output_left, output_top, output_right, output_bottom))

			# Save image
			# Crop image
			upscale_image = upscale_image.crop((0, 0, int(input_im.width * width_scale), int(input_im.height * height_scale)))
			upscale_image.save(args.save_path)


# If running the script as main executable
if __name__ == '__main__':
	super_resolution_upscale(sys.argv[1:])
