# !/usr/bin/env python3
import argparse
from datetime import date
import itertools
import os
import random
import sys

from SuperResolution import dcsuperresolution_program
from core.common import DefaultArgumentParser

parser = argparse.ArgumentParser(add_help=True, prog="SuperResolution Model Evolution",
								 description='Super Resolution Training Model Evolution Program',
								 parents=[DefaultArgumentParser()])

#
parser.add_argument('--output-dir', type=str, dest='output_dir',
					default=str.format(
						"super-resolution-evolution-{0}", date.today().strftime("%b-%d-%Y_%H:%M:%S")),
					help='Set the output directory that all the models and results will be stored at')
#
parser.add_argument('--models', dest='models', action='store', nargs='*', required=False,
					default=['srcnn', 'dcsr', 'edsr',
							 'dcsr-ae', 'dcsr-resnet', 'vdsr', 'srgan', 'esrgan'],
					choices=['srcnn', 'dcsr', 'edsr',
							 'dcsr-ae', 'dcsr-resnet', 'vdsr', 'srgan', 'esrgan'],
					help='Override what Model to include in training evolution.')
#
parser.add_argument('--loss-functions', dest='loss_functions', action='store', nargs='*', required=False,
					default=['mse', 'ssim', 'msa', 'vgg16', 'vgg19'],
					choices=['mse', 'ssim', 'msa', 'vgg16', 'vgg19'],
					help='Override what Loss functions to include in training evolution.')
#
parser.add_argument('--optimizer-evolution', dest='optimizer_evolution', action='append', nargs='*', required=False,
																				default=['adam',
																						 'rmsprop', 'sgd', 'adadelta'],
																				choices=['adam',
																						 'rmsprop', 'sgd', 'adadelta'],
																				help='Select optimizer to be used')

# If invalid number of arguments, print help.
if len(sys.argv) < 2:
	parser.print_help()
	sys.exit(1)

args = parser.parse_args(args=sys.argv[1:])

# Extract parameters
epochs: int = args.epochs
output_dir: str = args.output_dir
batch_size: int = args.batch_size
training_dataset_paths = args.train_directory_paths
validation_dataset_paths = args.validation_directory_paths
test_dataset_paths = args.test_directory_paths
input_image_size: tuple = args.input_image_size
image_output_size: tuple = args.output_image_size
seed: int = args.seed
models: list = args.models
loss_functions: list = args.loss_functions
optimizer: list = args.optimizer_evolution
usefp16: bool = args.use_float16
use_cpu_only: bool = args.use_explicit_cpu
metrics: str = "psnr ssim"

hyperparameters = {
	#
	"--optimizer": optimizer,
	"--color-space": ["rgb", "lab"],
	"--learning-rate": [0.0015, 0.0008, 0.0001, 0.0003, 0.002],
	"--example-batch": [512],
	"--example-batch-grid-size": [8],
	# TODO: add each model specific parameter options.
	"--regularization": [0.00001, 0.001, 0.002, 0.003, 0.008, 0.0],
	"--decay-rate": [0.85, 0.90, 0.96],
	"--decay-step": [1000, 10000, 50000],
	"--loss-fn": loss_functions,
	"--seed": [seed],
	"--model": models,
	"--cache-file": [
		"/tmp/super-resolution-cache-" + os.path.basename(os.path.normpath(str(output_dir)))],
	"--epochs": [epochs],
	"--shuffle-data-set-size": [1024],
	"--checkpoint-every-epoch": [0],
	"--data-set-directory": training_dataset_paths,
	"--batch-size": [batch_size],
}

if validation_dataset_paths:
	hyperparameters['--validation-data-directory'] = validation_dataset_paths

# Optionally add test.
if test_dataset_paths:
	hyperparameters['--test-set-directory'] = test_dataset_paths

#
keys, values = zip(*hyperparameters.items())
hyperparameter_combinations = [dict(zip(keys, v))
							   for v in itertools.product(*values)]

#
random.shuffle(hyperparameter_combinations)
print("Number of hyperparameter combinations: " +
	  str(len(hyperparameter_combinations)))

for i, custom_argv in enumerate(hyperparameter_combinations):

	argvlist = []
	for key, value in custom_argv.items():
		argvlist.append(str(key))
		argvlist.append(str(value))

	# Add input resolution.
	argvlist.append("--image-size")
	argvlist.append(str(input_image_size[0]))
	argvlist.append(str(input_image_size[1]))

	# Add output resolution
	argvlist.append("--output-image-size")
	argvlist.append(str(image_output_size[0]))
	argvlist.append(str(image_output_size[1]))

	output_target_dir = str(os.path.join(output_dir,
										 "It{0}Learning:{1}DecRate:{2}ColorSpace:{3}Loss:{4}Model:{5}".format(
											 i,
											 custom_argv["--learning-rate"],
											 custom_argv["--decay-rate"],
											 custom_argv["--color-space"],
											 custom_argv["--loss-fn"],
											 custom_argv["--model"])))
	# Add option
	argvlist.append("--output-dir")
	argvlist.append(output_target_dir)
	argvlist.append("--show-psnr")
	if use_cpu_only:
		argvlist.append("--cpu")
	if usefp16:
		argvlist.append("--use-float16")
	
	#
	argvlist.append("--metrics ")
	for m in metrics:
		argvlist.append(m)

	dcsuperresolution_program(argvlist)
