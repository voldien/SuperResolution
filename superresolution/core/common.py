import argparse
import logging
import os
from random import randrange

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.python.client import device_lib


def DefaultArgumentParser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(add_help=False)

	#
	parser.add_argument('--epochs', type=int, default=48, dest='epochs',
						help='Set the number of passes that the training set will be trained against.')
	#

	parser.add_argument('--batch-size', type=int, default=16, dest='batch_size',
						help='number of training element per each batch, during training.')
	#
	parser.add_argument('--checkpoint-filepath', type=str, dest='checkpoint_dir',
						default="./training_checkpoints",
						help='Set the path the checkpoint will be saved/loaded.')
	#
	parser.add_argument('--checkpoint-every-epoch', type=int, dest='checkpoint_every_nth_epoch',
						default=2,
						help='Set how often the checkpoint will be update, per epoch.')
	#
	parser.add_argument('--learning-rate', type=float,
						dest='learning_rate', default=0.0002, help='Set the initial Learning Rate')

	# Default, all devices will be used.
	parser.add_argument('--device', action='append', default=None, required=False,
						dest='devices', help='Select the device explicitly that will be used.')
	# TODO:extract str					choices=device_lib.list_local_devices())

	parser.add_argument('--cpu', action='store_true',
						default=False,
						dest='use_explicit_cpu', help='Explicit use the CPU as the compute device.')

	parser.add_argument('--gpu', action='store_true',
						default=None, required=False,
						dest='use_explicit_gpu', help='Explicit use of GPU')

	#
	parser.add_argument('--distribute-strategy', action='store', default=None,
						dest='distribute_strategy', help='Select Distribute Strategy.',
						choices=['mirror'])

	#
	parser.add_argument('--verbosity', type=int, dest='verbosity',
						default=logging.INFO,
						help='Set the verbosity level of the program')

	#
	parser.add_argument('--use-float16', action='store_true',
						dest='use_float16', default=False, help='Hint the usage of Float 16 (FP16) in the model.')
	#
	parser.add_argument('--cache-ram', action='store_true', default=False,
						dest='cache_ram', help='Use System Memory (RAM) as Cache storage.')
	#
	parser.add_argument('--cache-file', type=str,
						dest='cache_path', default=None,
						help='Set the cache file path that will be used to store dataset cached data.')
	#
	parser.add_argument('--shuffle-data-set-size', type=int,
						dest='dataset_shuffle_size', default=1024,
						help='Set the size of the shuffle buffer size, zero disables shuffling.')

	parser.add_argument('--data-set-directory', dest='train_directory_paths', type=str,
						action='append',
						help='Directory path where the images are located dataset images')

	parser.add_argument('--validation-data-directory', dest='validation_directory_paths', type=str,
						action='append',
						help='Directory path where the images are located dataset images')

	parser.add_argument('--test-data-directory', dest='test_directory_paths', type=str,
						action='append',
						help='Directory path where the images are located dataset images')

	#
	parser.add_argument('--image-size', type=int, dest='image_size',
						nargs=2,
						default=(128, 128),
						help='Set the size of the images in width and height for the model.')

	parser.add_argument('--output-image-size', type=int, dest='output_image_size',
						nargs=2, required=False,
						default=(256, 256),
						help='Set the size of the images in width and height for the model.')
	#
	parser.add_argument('--seed', type=int, default=randrange(10000000), dest='seed',
						help='Set the random seed')

	parser.add_argument('--nr_image_example_generate', type=int, default=16, dest='num_examples_to_generate',
						help='Number')
	#
	parser.add_argument('--color-space', type=str, default="rgb", dest='color_space', choices=['rgb', 'lab'],
						help='Select Color Space used in the model.')
	#
	parser.add_argument('--color-channels', type=int, default=3, dest='color_channels', choices=[1, 3, 4],
						help='Select Number of channels in the color space. GrayScale, RGB and RGBA.')
	#
	parser.add_argument('--optimizer', type=str, default='adam', dest='optimizer',
						choices=['adam', 'ada', 'rmsprop', 'sgd', 'adadelta'],
						help='Select optimizer to be used')

	parser.add_argument('--disable-validation', default=True, dest='use_validation', action='store_false',
						help='Select if use data validation step.')
	return parser


def ParseDefaultArgument(args: dict):
	#
	tf.config.experimental.enable_tensor_float_32_execution(True)
	# Set global precision default policy.
	if args.use_float16:
		mixed_precision.set_global_policy('mixed_float16')
	else:
		mixed_precision.set_global_policy('float32')

	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
		finally:
			pass

	# Allow device override if not present automatically.
	tf.config.set_soft_device_placement(True)

	# Create output directory if not exists.
	output_path = os.path.abspath(os.path.join(os.path.curdir, args.output_dir))
	if not os.path.exists(output_path):
		os.makedirs(output_path, exist_ok=True)

	# Establish the checkpoint directory.
	if not os.path.isabs(args.checkpoint_dir):
		args.checkpoint_dir = os.path.join(args.output_dir, args.checkpoint_dir)


def create_virtual_gpu_devices():
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		# Create 2 virtual GPUs with 1GB memory each
		try:
			tf.config.set_logical_device_configuration(
				gpus[0],
				[tf.config.LogicalDeviceConfiguration(memory_limit=1024),
				 tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
			return tf.config.list_logical_devices('GPU')

		except RuntimeError as e:
			# Virtual devices must be set before GPUs have been initialized
			print(e)
	return []
