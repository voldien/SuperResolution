# !/usr/bin/env python3
import argparse
import json
import logging
import os
import pathlib
import sys
import traceback
from importlib import import_module
from random import randrange
from typing import Dict

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_io as tfio

from core import ModelBase

import models.DCSuperResolution
import models.PostDCSuperResolution
import models.SuperResolutionAE
import models.SuperResolutionEDSR
import models.SuperResolutionResNet
import models.SuperResolutionVDSR
import models.SuperResolutionCNN

from core.CommandCommon import ParseDefaultArgument, DefaultArgumentParser
from util.dataProcessing import load_dataset_from_directory, \
	configure_dataset_performance, dataset_super_resolution, augment_dataset, split_dataset
from util.math import psnr
from util.metrics import PSNRMetric
from util.trainingcallback import GraphHistory, SaveExampleResultImageCallBack
from util.util import plotTrainingHistory

def setup_dataset(dataset, args: dict):
	pass

def create_setup_optimizer(args: dict):

	learning_rate: float = args.learning_rate
	learning_decay_step: int = args.learning_rate_decay_step
	learning_decay_rate: float = args.learning_rate_decay

	# Setup Learning Rate with Decay.
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		learning_rate,
		decay_steps=learning_decay_step,
		decay_rate=learning_decay_rate,
		staircase=False)

	#
	parameters = {'beta_1': 0.5, 'beta_2': 0.9, 'learning_rate': lr_schedule}
	return tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.9)
	# model_optimizer = tf.keras.optimizers.get(args.optimizer, kwargs=parameters)

	return model_optimizer

def setup_tensorflow_strategy(args: dict):
	# Configure
	if args.use_explicit_cpu:
		selected_devices = tf.config.list_logical_devices('CPU')
	else:
		selected_devices = tf.config.list_logical_devices('GPU')
		if args.devices is not None:
			# TODO add support
			selected_devices = args.devices
		else:
			pass

	# initialize
	strategy = tf.distribute.MirroredStrategy(devices=selected_devices)
	logger.info('Number of devices: {0}'.format(strategy.num_replicas_in_sync))
	return strategy

def load_builtin_model_interfaces() -> Dict[str, ModelBase]:
	builtin_models: Dict[str, ModelBase] = {}

	builtin_models['dcsr'] = models.DCSuperResolution.get_model_interface()
	builtin_models['dcsr-post'] = models.PostDCSuperResolution.get_model_interface()
	builtin_models['edsr'] = models.SuperResolutionEDSR.get_model_interface()
	builtin_models['dcsr-ae'] = models.SuperResolutionAE.get_model_interface()
	builtin_models['dcsr-resnet'] = models.SuperResolutionResNet.get_model_interface()
	builtin_models['vdsr'] = models.SuperResolutionVDSR.get_model_interface()
	builtin_models['cnnsr'] = models.SuperResolutionCNN.get_model_interface()

	return builtin_models

def load_model_interface(model_name: str) -> ModelBase:
	# Load ML model from either runtime from script or from builtin.
	builtin_models = load_builtin_model_interfaces()
	module_interface: ModelBase = None

	# Load model module from python file if provided.
	if os.path.isfile(model_name) and model_name.endswith('.py'):
		dynamic_module = import_module(model_name)
		module_interface = dynamic_module.get_model_interface()
	else:
		module_interface = builtin_models.get(model_name)

	return module_interface


def setup_model(args: dict, builtin_models: Dict[str, ModelBase], image_input_size: tuple,
				image_output_size: tuple) -> keras.Model:
	
	args.model_override_filepath = None  # TODO remove
	if args.model_override_filepath is not None:
		return tf.keras.models.load_model(
			args.model_override_filepath, compile=False)
	else:

		model_name: str = args.model
		# Load ML model from either runtime from script or from builtin.
		module_interface = load_model_interface(model_name)

		# Must be a valid interface.
		if module_interface is None:
			raise RuntimeError("Could not find model interface: " + model_name)

		return module_interface.create_model(input_shape=image_input_size, output_shape=image_output_size, kwargs=args)

def setup_loss_builtin_function(args: dict):
	def ssim_loss(y_true, y_pred):
		# TODO convert color space.
		y_true_color = None
		y_pred_color = None

		if args.color_space == 'rgb':
			# Remape [-1,1] to [0,1]
			y_true_color = ((y_true + 1.0) * 0.5)
			y_pred_color = ((y_pred + 1.0) * 0.5)
		elif args.color_space == 'lab':
			# Remape [-1,1] -> [-128, 128] -> [0,1]
			y_true_color = tfio.experimental.color.lab_to_rgb(y_true * 128)
			y_pred_color = tfio.experimental.color.lab_to_rgb(y_pred * 128)
		else:
			assert 0

		return (1 - tf.reduce_mean(tf.image.ssim(y_true_color, y_pred_color, max_val=1.0, filter_size=11,
												 filter_sigma=1.5, k1=0.01, k2=0.03)))

	def pnbr_loss(y_true, y_pred):
		return 1 - psnr(y_true, y_pred)

	#
	builtin_loss_functions = {'mse': tf.keras.losses.MeanSquaredError(), 'ssim': ssim_loss,
							  'msa': tf.keras.losses.MeanAbsoluteError(), 'pnbr': pnbr_loss}

	return builtin_loss_functions[args.loss_fn]


def run_train_model(args: dict, dataset):
	# Configure how models will be executed.
	strategy = setup_tensorflow_strategy(args=args)

	# Compute the total batch size.
	batch_size: int = args.batch_size * strategy.num_replicas_in_sync
	logger.info("Number of batches {0} of {1} elements".format(
		len(dataset), batch_size))

	# Create Input and Output Size
	# TODO determine size.
	image_input_size = (
		int(args.image_size[0] / 2), int(args.image_size[1] / 2), args.color_channels)
	image_output_size = (
		args.image_size[0], args.image_size[1], args.color_channels)

	logging.info("Input Size {0}".format(image_input_size))
	logging.info("Output Size {0}".format(image_output_size))

	# Setup none-augmented version for presentation.
	non_augmented_dataset_train = dataset_super_resolution(dataset=dataset,
														   input_size=image_input_size,
														   output_size=image_output_size)

	non_augmented_dataset_train = tf.data.Dataset.zip((non_augmented_dataset_train, non_augmented_dataset_train))
	non_augmented_dataset_train = dataset.batch(batch_size)
	non_augmented_dataset_train, non_augmented_dataset_validation = split_dataset(dataset=dataset, train_size=0.95)

	# Configure cache, shuffle and performance of the dataset.
	dataset = configure_dataset_performance(ds=dataset, use_cache=args.cache_ram, cache_path=args.cache_path,
											shuffle_size=args.dataset_shuffle_size)

	# Apply data augmentation
	dataset = augment_dataset(dataset=dataset, image_crop_shape=image_output_size)

	# Transform data to fit upscale.
	dataset = dataset_super_resolution(dataset=dataset,
									   input_size=image_input_size,
									   output_size=image_output_size)

	# Final Combined dataset. Set batch size
	dataset = dataset.batch(batch_size)

	# Split training and validation.
	dataset, validation_data_ds = split_dataset(dataset=dataset, train_size=0.95)

	# Setup for strategy support, to allow multiple GPU setup.
	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
	dataset = dataset.with_options(options)

	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
	validation_data_ds = validation_data_ds.with_options(options)

	# Setup builtin models
	builtin_models = load_builtin_model_interfaces()

	#
	logging.info(len(dataset))

	#
	with strategy.scope():
		# Creating optimizer.
		model_optimizer = create_setup_optimizer(args=args)

		# Load or create models.
		training_model = setup_model(args=args, builtin_models=builtin_models, image_input_size=image_input_size,
									 image_output_size=image_output_size)

		# Save the model as an image to directory, for easy backtracking of the model composition.
		tf.keras.utils.plot_model(
			training_model, to_file=os.path.join(args.output_dir, 'Model.png'),
			show_shapes=True, show_dtype=True,
			show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
			layer_range=None
		)

		logger.debug(training_model.summary())

		# NOTE currently, only support checkpoint if generated model and not when using existing.
		loss_fn = setup_loss_builtin_function(args)

		# TODO metric list.
		metrics = ['accuracy', ]
		if args.show_psnr:
			metrics.append(PSNRMetric())

		training_model.compile(optimizer=model_optimizer, loss=loss_fn, metrics=metrics)
		# Save copy.
		training_model.save(args.model_filepath)

		# Create a callback that saves the model's.py weights
		checkpoint_path: str = args.checkpoint_dir

		# Create a callback that saves the model's.py weights
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(checkpoint_path + "-{epoch:02d}.hdf5"),
														 monitor='val_loss',
														 save_weights_only=True,
														 save_freq='epoch',
														 verbose=0)

		# If exists, load weights.
		if os.path.exists(checkpoint_path):
			training_model.load_weights(checkpoint_path)

		# TODO add none-augmented dataset. to see progress better.
		#
		args.example_batch_grid_size

		ExampleResultCallBack = SaveExampleResultImageCallBack(
			args.output_dir,
			dataset.take(
				32), args.color_space)
		graph_output_filepath: str = os.path.join(args.output_dir, "history_graph.png")
		graphHistory = GraphHistory(filepath=graph_output_filepath)

		# TODO
		validation_data_ds = None
		historyResult = training_model.fit(x=dataset, validation_data=validation_data_ds, verbose='auto',
										   epochs=args.epochs,
										   callbacks=[cp_callback, tf.keras.callbacks.TerminateOnNaN(),
													  ExampleResultCallBack,
													  graphHistory])

		# Save final model.
		training_model.save(args.model_filepath)

		# Plot history result.
		plotTrainingHistory(historyResult.history)


def dcsuperresolution_program(vargs=None):
	try:
		# Create logger and output information
		global logger
		logger = logging.getLogger("Super Resolution Logger")

		parser = argparse.ArgumentParser(
			prog='SuperResolution',
			add_help=True,
			description='Super Resolution Training Model Program',
			parents=[DefaultArgumentParser()]
		)

		# Model Save Path.
		default_generator_id = randrange(10000000)
		parser.add_argument('--model-filename', type=str, dest='model_filepath',
							default=str.format(
								"super-resolution-model-{0}.h5", default_generator_id),
							help='Define file path that the generator model will be saved at.')
		#
		parser.add_argument('--output-dir', type=str, dest='output_dir',
							default="",
							help='Set the output directory that all the models and results will be stored at')

		#
		parser.add_argument('--example-batch', dest='example_batch',  # TODO rename
							type=int,
							default=1024,
							help='Set the number of train batches between saving work in progress result.')
		#
		parser.add_argument('--example-batch-gridsize', dest='example_batch_grid_size',
							type=int, metavar=('width', 'height'),
							nargs=2, default=(8, 8), help='Set the grid size of number of example images.')

		#
		parser.add_argument('--show-psnr', dest='show_psnr',
							type=bool,
							nargs=1, default=False, help='Set the grid size of number of example images.')

		# TODO add support
		parser.add_argument('--metrics', dest='metrics',
							type=str, action='append',
							nargs=1, default=False, help='Set what metric to capture.')

		#
		parser.add_argument('--decay-rate', dest='learning_rate_decay',
							default=0.98,
							help='Set Learning rate Decay.', type=float)
		#
		parser.add_argument('--decay-step', dest='learning_rate_decay_step',
							default=40000,
							help='Set Learning rate Decay Step.', type=int)
		#
		parser.add_argument('--model', dest='model',
							default='dcsr',
							choices=['cnnsr', 'dcsr', 'dscr-post', 'dscr-pre', 'edsr', 'dcsr-ae','dcsr-resnet','vdsr'],
							help='Set which model type to use.', type=str)
		#
		parser.add_argument('--loss-fn', dest='loss_fn',
							default='mse',
							choices=['mse', 'ssim', 'msa', 'pnbr', 'vgg16', 'none'],
							help='Set Loss Function', type=str)

		# Load model and iterate existing models.
		model_list = load_builtin_model_interfaces()

		for model_object_inter in model_list.values():
			logger.info("Found Model: " + model_object_inter.get_name())
			parser.add_argument_group(model_object_inter.get_name(), model_object_inter.load_argument())

		# If invalid number of arguments, print help.
		if len(sys.argv) < 2:
			parser.print_usage()
			sys.exit(1)

		args = parser.parse_args(args=vargs)

		# Parse for common arguments.
		ParseDefaultArgument(args)

		# Set init seed.
		tf.random.set_seed(args.seed)

		# Setup logging
		logger.setLevel(args.verbosity)

		#
		console_handler = logging.StreamHandler()
		
		#
		log_format = '%(asctime)s | %(levelname)s: %(message)s'
		console_handler.setFormatter(logging.Formatter(log_format))

		#
		logger.addHandler(console_handler)
		logger.addHandler(logging.FileHandler(filename=os.path.join(args.output_dir, "log.txt")))

		# Logging about all the options etc.
		logger.info(str.format("Epochs: {0}", args.epochs))
		logger.info(str.format("Batch Size: {0}", args.batch_size))

		logger.info(str.format("Use float16: {0}", args.use_float16))

		logger.info(str.format("CheckPoint Save Every Nth Epoch: {0}", args.checkpoint_every_nth_epoch))

		logger.info(str.format("Use RAM Cache: {0}", args.cache_ram))

		logger.info(str.format("Example Batch Grid Size: {0}", args.example_batch_grid_size))
		logger.info(str.format("Image Training Set: {0}", args.image_size))
		logger.info(str.format("Learning Rate: {0}", args.learning_rate))
		logger.info(str.format(
			"Learning Decay Rate: {0}", args.learning_rate_decay))
		logger.info(str.format(
			"Learning Decay Step: {0}", args.learning_rate_decay_step))
		logger.info(str.format("Image ColorSpace: {0}", args.color_space))
		logger.info(str.format("Output Directory: {0}", args.output_dir))

		# Create absolute path for model file, if relative path.
		if not os.path.isabs(args.model_filepath):
			args.model_filepath = os.path.join(args.output_dir, args.model_filepath)

		# Allow override to enable cropping for increase details in the dataset.
		override_size: tuple = (512, 512)
		data_set_filepaths = args.data_sets_directory_paths

		# Setup Dataset
		dataset = None
		if data_set_filepaths:
			for directory_path in data_set_filepaths:
				# Check if directory.
				if os.path.isdir(directory_path):
					pass

				data_dir = pathlib.Path(directory_path)
				logging.info("Loading dataset directory {0}".format(data_dir))

				local_dataset = load_dataset_from_directory(data_path=data_dir, args=args, override_size=override_size)
				if not dataset:
					dataset = local_dataset
				else:
					dataset.concatenate(local_dataset)

		if not dataset:
			logger.error("Failed to construct dataset")
			raise RuntimeError("Could not create dataset from {0}".format(data_set_filepaths))

		# Make a copy of the command line.
		commandline = ' '.join(vargs)
		commandline_filepath = os.path.join(args.output_dir, "commandline")
		with open(commandline_filepath, 'w') as writefile:
			writefile.write(commandline)

		# Save the argument options to file, for backtracking of the hyperparameters of the training.
		config_filepath = os.path.join(args.output_dir, "config_options.json")
		with open(config_filepath, 'w') as writefile:
			json.dump(args.__dict__, writefile, indent=2)

		# Main Train Model
		run_train_model(args, dataset)

	except Exception as ex:
		print(ex)
		logger.error(ex)

		traceback.print_exc()


# If running the script as main executable


if __name__ == '__main__':
	dcsuperresolution_program(vargs=sys.argv[1:])
