# !/usr/bin/env python3
import argparse
from random import randrange
import sys

from PIL import Image
import tensorflow as tf
import numpy as np
from skimage.color import lab2rgb, rgb2lab
import tensorflow_io as tfio
import sys
import cv2
from util.image import generate_grid_image
from numpy import asarray
from random import randrange
import tensorflow as tf
import numpy as np
import imageio.v2 as imageio
from skimage.color import lab2rgb
import os.path
import logging
import argparse
import os
from tensorflow.python.client import device_lib


def upscale_image(model, image) -> Image:
	result_upscale =  model(image, Training= False)
	#decoder_image = model.predict(cropped_sub_image,verbose=0) * 128.0
	decoder_image = np.asarray(lab2rgb(result_upscale[0])).astype(dtype='float32')

	decoder_image_u8 = np.uint8(result_upscale * 255)
	compressed_crop_im = Image.fromarray(result_upscale, "RGB")

	return compressed_crop_im


def generate_transition_program(argv):

	parser = argparse.ArgumentParser(
		description='GAN (Generative Adversarial Networks) Latent Space Viewer')

	parser.add_argument('--save-output', dest='save_path', default=None, help='')
	parser.add_argument('--model', dest='model', default=None, help='')


	parser.add_argument('--device', type=str, dest='', default=None, help='')

	parser.add_argument('--verbosity', type=int, dest='accumulate',
	                    default=1,
	                    help='Define the save/load model path')

	parser.add_argument('--image-filter', type=tuple, dest='accumulate',
	                    default="*",
	                    help='Define the save/load model path')

	parser.add_argument('--seed', type=int, default=randrange(10000000), dest='seed',
	                    help='Define the save/load model path')

	parser.add_argument('--data-set-directory', type=str, dest='data_sets_directory_paths',
	                    #	                    action='append', nargs='*',
	                    # nargs=1,
	                    help='Directory path where the images are located dataset images')

	parser.add_argument('--color-space', type=str, default="rgb", dest='color_space', choices=['rgb', 'lab'],
	                    help='Select Color Space Image wisll be decode from the output model data.')
	#
	parser.add_argument('--color-channels', type=int, default=3, dest='color_channels', choices=[1, 3, 4],
	                    help='Select Number of channels in the color space. GrayScale, RGB and RGBA.')

	args = parser.parse_args(args=argv)

	logger = logging.getLogger('latent space')
	logger.setLevel(logging.INFO)


	with tf.device('/device:CPU:0'):

		# Open files, convert to rgb.
		im = Image.open(sys.argv[1])
		im = im.convert('RGB')

		upscale_model = tf.keras.models.load_model(args.model)

		upscale_model.summary()

		latent_space_size = upscale_model.input_shape[1]
		input_width, input_height, input_channels = latent_space_size


		latent_values = []
		upscale_image = Image.new("RGB", im.size, im.getpixel((0, 0)))

		for x in range(0, int(im.width / input_width)):
			for y in range(0, int(im.height / input_height)):

				# Compute subset view.
				left = x * input_width
				top = y * input_height
				right = (x + 1) * input_width
				bottom = (y + 1) * input_height

				# Create subset image.
				cropped_sub_image = im.crop((left, top, right, bottom))

				# 
				cropped_sub_image = rgb2lab((np.array(cropped_sub_image) * (1.0 / 255.0)).astype(dtype='float32')) * (1.0 / 128.0)
				cropped_sub_image = np.expand_dims(cropped_sub_image, axis=0)

				upscale_result = upscale_image(upscale_model, cropped_sub_image)

				decoder_image = np.asarray(lab2rgb(decoder_image[0])).astype(dtype='float32')

				upscale_image.paste(upscale_result, (left, top, right, bottom))

		# Save image
		upscale_image.save(args.save_path)

# If running the script as main executable
if __name__ == '__main__':
	generate_transition_program(sys.argv[1:])
