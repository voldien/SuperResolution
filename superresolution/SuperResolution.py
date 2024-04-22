# !/usr/bin/env python3
import argparse
from datetime import date
import json
import logging
import os
import pathlib
import sys
import traceback
from importlib import import_module
from logging import Logger
from random import randrange
from typing import Dict
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from tensorflow.python.data import Dataset

from core import ModelBase

import models.DCSuperResolution
import models.PostDCSuperResolution
import models.SuperResolutionAE
import models.SuperResolutionEDSR
import models.SuperResolutionResNet
import models.SuperResolutionVDSR
import models.SuperResolutionCNN
import models.SuperResolutionSRGAN
import models.SuperResolutionESRGAN

from core.common import ParseDefaultArgument, DefaultArgumentParser, setup_tensorflow_strategy
from util.dataProcessing import load_dataset_from_directory, \
	configure_dataset_performance, dataset_super_resolution, augment_dataset
from util.metrics import PSNRMetric, SSIMMetric
from util.trainingcallback import GraphHistory, SaveExampleResultImageCallBack, \
	CompositeImageResultCallBack
from util.util import plotTrainingHistory
from util.loss import PSNRError, SSIMError, VGG16Error, VGG19Error

global sr_logger
sr_logger: Logger = logging.getLogger("Super Resolution Training")


def setup_dataset(dataset, args: dict):
	pass


def create_metric(metrics: list):
	metric = []
	for m in metrics:
		if m == 'ssim':
			metric.append(SSIMMetric())
		elif m == 'psnr':
			metric.append(PSNRMetric())
	return metric


def create_setup_optimizer(args: dict):
	learning_rate: float = args.learning_rate
	learning_decay_step: int = args.learning_rate_decay_step
	learning_decay_rate: float = args.learning_rate_decay
	optimizer: str = args.optimizer

	# Setup Learning Rate with Decay.
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		learning_rate,
		decay_steps=learning_decay_step,
		decay_rate=learning_decay_rate,
		staircase=False)

	#
	if optimizer == 'adam':
		return tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.9)
	elif optimizer == 'rmsprop':
		return tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
	elif optimizer == 'sgd':
		return tf.keras.optimizers.SGD(learning_rate=lr_schedule)
	elif optimizer == 'adadelta':
		return tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)
	else:
		raise ValueError(optimizer + " Is not a valid option")


def load_dataset_collection(filepaths: list, args: dict, override_size: tuple) -> Dataset:
	# Setup Dataset
	training_dataset = None
	for directory_path in filepaths:
		# Check if directory.
		if os.path.isdir(directory_path):
			pass

		data_dir = pathlib.Path(directory_path)
		logging.info("Loading dataset directory {0}".format(data_dir))

		local_dataset = load_dataset_from_directory(
			data_path=data_dir, args=args, override_size=override_size)
		if not training_dataset:
			training_dataset = local_dataset
		else:
			training_dataset.concatenate(local_dataset)

	return training_dataset


def load_builtin_model_interfaces() -> Dict[str, ModelBase]:
	builtin_models: Dict[str, ModelBase] = {}

	builtin_models['dcsr'] = models.DCSuperResolution.get_model_interface()
	builtin_models['dcsr-post'] = models.PostDCSuperResolution.get_model_interface()
	builtin_models['edsr'] = models.SuperResolutionEDSR.get_model_interface()
	builtin_models['dcsr-ae'] = models.SuperResolutionAE.get_model_interface()
	builtin_models['dcsr-resnet'] = models.SuperResolutionResNet.get_model_interface()
	builtin_models['vdsr'] = models.SuperResolutionVDSR.get_model_interface()
	builtin_models['srcnn'] = models.SuperResolutionCNN.get_model_interface()
	builtin_models['srgan'] = models.SuperResolutionSRGAN.get_model_interface()
	builtin_models['esrgan'] = models.SuperResolutionESRGAN.get_model_interface()

	return builtin_models


def load_model_interface(model_name: str) -> ModelBase:
	# Load ML model from either runtime from script or from builtin.
	builtin_models = load_builtin_model_interfaces()

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

		# Must be a valid interface.pk
		if module_interface is None:
			raise RuntimeError("Could not find model interface: " + model_name)

		argument_dic = vars(args)
		return module_interface.create_model(input_shape=image_input_size, output_shape=image_output_size,
		                                     **argument_dic)


def setup_loss_builtin_function(args: dict):
	#
	builtin_loss_functions = {'mse': tf.keras.losses.MeanSquaredError(),
	                          'ssim': SSIMError(color_space=args.color_space),
	                          'msa': tf.keras.losses.MeanAbsoluteError(), 'psnr': PSNRError(), 'vgg16': VGG16Error(),
	                          'vgg19': VGG19Error()}

	return builtin_loss_functions[args.loss_fn]


def run_train_model(args: dict, training_dataset: Dataset, validation_dataset: Dataset = None,
                    test_dataset: Dataset = None):
	# Configure how models will be executed.
	strategy = setup_tensorflow_strategy(args=args)
	sr_logger.info('Number of devices: {0}'.format(
		strategy.num_replicas_in_sync))

	# Compute the total batch size.
	batch_size: int = args.batch_size * strategy.num_replicas_in_sync
	sr_logger.info("Number of batches {0} of {1} elements".format(
		len(training_dataset), batch_size))

	# Create Input and Output Size
	image_input_size = (
		int(args.input_image_size[0]), int(args.input_image_size[1]),
		args.color_channels)
	image_output_size = (
		args.output_image_size[0], args.output_image_size[1], args.color_channels)

	logging.info("Input Size {0}".format(image_input_size))
	logging.info("Output Size {0}".format(image_output_size))

	# Setup none-augmented version for presentation.
	non_augmented_dataset_train = dataset_super_resolution(dataset=training_dataset,
	                                                       input_size=image_input_size,
	                                                       output_size=image_output_size, crop=True)
	non_augmented_dataset_validation = None
	if validation_dataset:
		non_augmented_dataset_validation = dataset_super_resolution(dataset=validation_dataset,
		                                                            input_size=image_input_size,
		                                                            output_size=image_output_size)
		non_augmented_dataset_validation = non_augmented_dataset_validation.batch(
			batch_size)

		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		non_augmented_dataset_validation = non_augmented_dataset_validation.with_options(
			options)

	non_augmented_dataset_train = configure_dataset_performance(ds=non_augmented_dataset_train, use_cache=False,
	                                                            cache_path=None, shuffle_size=0)

	non_augmented_dataset_train = non_augmented_dataset_train.batch(batch_size)

	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
	non_augmented_dataset_train = non_augmented_dataset_train.with_options(
		options)

	# Configure cache, shuffle and performance of the dataset.
	training_dataset = configure_dataset_performance(ds=training_dataset, use_cache=args.cache_ram,
	                                                 cache_path=args.cache_path,
	                                                 shuffle_size=args.dataset_shuffle_size)

	# Apply data augmentation
	training_dataset = augment_dataset(
		dataset=training_dataset, image_crop_shape=image_output_size)

	# Transform data to fit upscale.
	training_dataset = dataset_super_resolution(dataset=training_dataset,
	                                            input_size=image_input_size,
	                                            output_size=image_output_size)

	# Final Combined dataset. Set batch size
	training_dataset = training_dataset.batch(batch_size)

	# Setup for strategy support, to allow multiple GPU setup.
	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
	training_dataset = training_dataset.with_options(options)

	# Split training and validation.
	validation_data_ds = None
	if args.use_validation and validation_dataset:
		# Configure cache, shuffle and performance of the dataset.
		validation_data_ds = configure_dataset_performance(ds=validation_dataset, use_cache=False,
		                                                   cache_path=None,
		                                                   shuffle_size=0)
		# Apply data augmentation
		validation_data_ds = augment_dataset(
			dataset=validation_data_ds, image_crop_shape=image_output_size)

		# Transform data to fit upscale.
		validation_data_ds = dataset_super_resolution(dataset=validation_data_ds,
		                                              input_size=image_input_size,
		                                              output_size=image_output_size)
		validation_data_ds = validation_data_ds.batch(batch_size)

		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		validation_data_ds = validation_data_ds.with_options(options)

	# Setup builtin models
	builtin_models = load_builtin_model_interfaces()

	#
	logging.info(len(training_dataset))

	#
	with strategy.scope():
		# Creating optimizer.
		model_optimizer = create_setup_optimizer(args=args)

		# Load or create models.
		training_model = setup_model(args=args, builtin_models=builtin_models, image_input_size=image_input_size,
		                             image_output_size=image_output_size)

		sr_logger.debug(training_model.summary())

		#
		loss_fn = setup_loss_builtin_function(args)

		# Metric list.
		metrics = [tf.keras.metrics.Accuracy(), ]
		additional_metrics = create_metric(args.metrics)
		if additional_metrics:
			metrics.extend(additional_metrics)

		training_model.compile(optimizer=model_optimizer,
		                       loss=loss_fn, metrics=metrics)

		# Save the model as an image to directory, for easy backtracking of the model composition.
		tf.keras.utils.plot_model(
			training_model, to_file=os.path.join(args.output_dir, 'Model.png'),
			show_shapes=True, show_dtype=True,
			show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
			layer_range=None
		)

		# checkpoint root_path
		checkpoint_root_path: str = args.checkpoint_dir

		# TODO: improve
		if os.path.exists(checkpoint_root_path):
			custom_objects = {
				'PSNRMetric': PSNRMetric(), 'VGG16Error': VGG16Error()}
			training_model = tf.keras.models.load_model(
				checkpoint_root_path, custom_objects=custom_objects)

		training_callbacks: list = [tf.keras.callbacks.TerminateOnNaN()]

		if args.use_checkpoint:
			# Create a callback that saves the model weights
			checkpoint_path = os.path.join(
				checkpoint_root_path, "cpkt-{epoch:02d}")
			checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
				filepath=checkpoint_path,
				monitor='val_loss' if validation_data_ds else 'loss',
				mode='min',
				save_freq='epoch',
				verbose=1)

			training_callbacks.append(checkpoint_callback)

		example_result_call_back = SaveExampleResultImageCallBack(
			args.output_dir,
			non_augmented_dataset_train, args.color_space,
			nth_batch_sample=args.example_nth_batch, grid_size=args.example_nth_batch_grid_size)
		training_callbacks.append(example_result_call_back)

		composite_train_callback = CompositeImageResultCallBack(
			dir_path=args.output_dir,
			name="train",
			train_data_subset=non_augmented_dataset_train, color_space=args.color_space)
		training_callbacks.append(composite_train_callback)

		# if non_augmented_dataset_validation:
		if non_augmented_dataset_validation:
			composite_validation_callback = CompositeImageResultCallBack(
				dir_path=args.output_dir,
				name="validation",
				train_data_subset=non_augmented_dataset_validation, color_space=args.color_space)
			training_callbacks.append(composite_validation_callback)

		graph_output_filepath: str = os.path.join(
			args.output_dir, "history_graph.png")
		training_callbacks.append(GraphHistory(filepath=graph_output_filepath))

		# Save copy.
		training_model.save(args.model_filepath)

		history_result = training_model.fit(x=training_dataset, validation_data=validation_data_ds, verbose='auto',
		                                    epochs=args.epochs,
		                                    callbacks=training_callbacks)
		#
		training_model.save(args.model_filepath)

		# Test model.
		if test_dataset:
			training_model.test(x=test_dataset, verbose='auto')

		# Plot history result.
		fig = plotTrainingHistory(history_result.history)
		fig.savefig(os.path.join(args.output_dir, "final_history_graph.png"))
		plt.close()


def dcsuperresolution_program(vargs=None):
	try:
		# Create logger and output information

		sr_logger.setLevel(level=logging.INFO)
		#
		console_handler = logging.StreamHandler()
		log_format = '%(asctime)s | %(levelname)s: %(message)s'
		console_handler.setFormatter(logging.Formatter(log_format))

		sr_logger.addHandler(console_handler)

		# Load model and iterate existing models.
		model_list: list[ModelBase] = load_builtin_model_interfaces()

		child_parsers: list = [DefaultArgumentParser()]

		parser = argparse.ArgumentParser(
			prog='SuperResolution',
			add_help=True,
			description='Super Resolution Training Model Program',
			parents=child_parsers
		)

		# Model Save Path.
		default_generator_id = randrange(10000000)
		parser.add_argument('--model-filename', type=str, dest='model_filepath', required=False,
		                    default=str.format(
			                    "super-resolution-model-{0}.keras", default_generator_id),
		                    help='Define file path that the generator model will be saved at.')
		#
		parser.add_argument('--output-dir', type=str, dest='output_dir',
		                    default=str.format(
			                    "super-resolution-{0}", date.today().strftime("%b-%d-%Y_%H:%M:%S")),
		                    help='Set the output directory that all the models and results will be stored at')
		#
		parser.add_argument('--example-batch', dest='example_nth_batch', required=False,  # TODO rename
		                    type=int,
		                    default=1024,
		                    help='Set the number of train batches between saving work in progress result.')
		#
		parser.add_argument('--example-batch-grid-size', dest='example_nth_batch_grid_size',
		                    type=int, required=False,
		                    default=8, help='Set the grid size of number of example images.')

		parser.add_argument('--metrics', dest='metrics',
		                    action='append',
		                    choices=['psnr', 'ssim'],
		                    default="", help='Set what metric to capture.')

		#
		parser.add_argument('--decay-rate', dest='learning_rate_decay',
		                    default=0.98, required=False,
		                    help='Set Learning rate Decay.', type=float)
		#
		parser.add_argument('--decay-step', dest='learning_rate_decay_step',
		                    default=10000, required=False,
		                    help='Set Learning rate Decay Step.', type=int)
		#
		parser.add_argument('--model', dest='model',
		                    default='dcsr',
		                    choices=['srcan', 'dcsr', 'dscr-post', 'dscr-pre', 'edsr', 'dcsr-ae', 'dcsr-resnet',
		                             'vdsr', 'srgan', 'esrgan'],
		                    help='Set which model type to use.', type=str)
		#
		parser.add_argument('--loss-fn', dest='loss_fn',
		                    default='mse',
		                    choices=['mse', 'ssim', 'msa',
		                             'psnr', 'vgg16', 'vgg19', 'none'],
		                    help='Set Loss Function', type=str)
		parser.add_argument('--override-image-size', type=int, dest='override_image_size',
		                    nargs=2, required=False,
		                    default=(-1, -1),
		                    help='')

		# If invalid number of arguments, print help.
		if len(sys.argv) < 2:
			parser.print_help()
			for model_object_inter in model_list.values():
				model_object_inter.load_argument().print_help()
			sys.exit(1)

		# Parse argument.
		args, _ = parser.parse_known_args(args=vargs)

		config_args = None
		if args.config:
			config_args = json.loads(args.config)
		# Parse for common arguments.
		ParseDefaultArgument(args)

		for model_object_inter in model_list.values():
			sr_logger.info("Found Model: " + model_object_inter.get_name())

		# Parse model specific argument parser.
		model_container = model_list[args.model]
		model_args, _ = model_container.load_argument().parse_known_args(args=vargs)
		# Add model argument result to the main argument result.

		# Merge namespace
		model_dic = vars(model_args)
		args_dic = vars(args)
		args_dic.update(model_dic)
		if config_args:
			args_dic.update(config_args)
		args = argparse.Namespace(**args_dic)

		# Set init seed.
		tf.random.set_seed(args.seed)

		# Override the default logging level
		sr_logger.setLevel(args.verbosity)
		# Add logging output path.
		sr_logger.addHandler(logging.FileHandler(
			filename=os.path.join(args.output_dir, "log.txt")))

		# Logging about all the options etc.
		sr_logger.info(str.format("Epochs: {0}", args.epochs))
		sr_logger.info(str.format("Batch Size: {0}", args.batch_size))

		sr_logger.info(str.format("Use float16: {0}", args.use_float16))

		sr_logger.info(str.format(
			"CheckPoint Save Every Nth Epoch: {0}", args.checkpoint_every_nth_epoch))

		sr_logger.info(str.format("Use RAM Cache: {0}", args.cache_ram))

		sr_logger.info(str.format(
			"Example Batch Grid Size: {0}", args.example_nth_batch_grid_size))
		sr_logger.info(str.format(
			"Image Training Set: {0}", args.input_image_size))
		sr_logger.info(str.format("Learning Rate: {0}", args.learning_rate))
		sr_logger.info(str.format(
			"Learning Decay Rate: {0}", args.learning_rate_decay))
		sr_logger.info(str.format(
			"Learning Decay Step: {0}", args.learning_rate_decay_step))
		sr_logger.info(str.format("Image ColorSpace: {0}", args.color_space))
		sr_logger.info(str.format("Output Directory: {0}", args.output_dir))

		# Create absolute path for model file, if relative path.
		if not os.path.isabs(args.model_filepath):
			args.model_filepath = os.path.join(
				args.output_dir, args.model_filepath)

		# Allow override to enable cropping for increase details in the dataset.
		override_image_size = args.override_image_size
		if override_image_size[0] > 0 and override_image_size[1] > 0:
			override_size = override_image_size  # TODO rename
		else:
			override_size: tuple = (args.output_image_size[0] * 4, args.output_image_size[1] * 4)

		# Setup Dataset
		training_dataset = None
		data_set_filepaths = args.train_directory_paths
		if data_set_filepaths:
			training_dataset = load_dataset_collection(filepaths=data_set_filepaths, args=args,
			                                           override_size=override_size)

		validation_dataset = None
		validation_set_filepaths = args.validation_directory_paths
		if validation_set_filepaths:
			validation_dataset = load_dataset_collection(filepaths=validation_set_filepaths, args=args,
			                                             override_size=override_size)

		test_dataset = None
		test_set_filepaths = args.test_directory_paths
		if test_set_filepaths:
			test_dataset = load_dataset_collection(
				filepaths=test_set_filepaths, args=args, override_size=override_size)

		if not training_dataset:
			sr_logger.error("Failed to construct dataset")
			raise RuntimeError(
				"Could not create dataset from {0}".format(data_set_filepaths))

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
		run_train_model(args, training_dataset,
		                validation_dataset, test_dataset)

	except Exception as ex:
		print(ex)
		sr_logger.error(ex)

		traceback.print_exc()


# If running the script as main executable
if __name__ == '__main__':
	dcsuperresolution_program(vargs=sys.argv[1:])
