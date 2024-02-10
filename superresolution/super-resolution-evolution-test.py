# !/usr/bin/env python3
import argparse
import itertools
import os
import random
import sys

from SuperResolution import dcsuperresolution_program

# Epochs, dataset, batch, size, imagesize, output dir
parser = argparse.ArgumentParser(add_help=True, prog="", description="")

#
parser.add_argument('--epochs', type=int, default=12, dest='epochs',
					help='Set the number of passes that the training set will be trained against.')
#
parser.add_argument('--batch-size', type=int, default=16, dest='batch_size',
					help='number of training element per each batch, during training.')
#
parser.add_argument('--data-set-directory', type=str, dest='train_directory_paths',
					action='append', required=True,
					help='Directory path where the images are located dataset images')
#
parser.add_argument('--validation-data-directory', dest='validation_directory_paths', type=str,
					action='append',
					help='Directory path where the images are located dataset images')
#
parser.add_argument('--test-data-directory', dest='test_directory_paths', type=str,
					action='append',
					help='Directory path where the images are located dataset images')
#
parser.add_argument('--output-dir', type=str, dest='output_dir',
					default="test_output",
					help='Set the output directory that all the models and results will be stored at')
#
parser.add_argument('--image-size', type=int, dest='image_size',
					nargs=2,
					default=(128, 128),
					help='Set the size of the images in width and height for the model.')
#
parser.add_argument('--output-image-size', type=int, dest='output_image_size',
					nargs=2,
					default=(256, 256),
					help='Set the size of the images in width and height for the model.')
#
parser.add_argument('--model', dest='model', action='append', nargs='*', required=False,
					default=['cnnsr', 'dcsr', 'edsr', 'dcsr-ae', 'dcsr-resnet', 'vdsr'],
					choices=['cnnsr', 'dcsr', 'edsr', 'dcsr-ae', 'dcsr-resnet', 'vdsr'],
					help='Overide what Model to include in training evolution.')
#
parser.add_argument('--seed', type=int, dest='seed',
					nargs=1,
					default=random.randrange(10000000),
					help='Seed')
parser.add_argument('--cpu', action='store_true',
					default=False,
					dest='use_explicit_cpu', help='Explicit use the CPU as the compute device.')



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
image_size: tuple = args.image_size
image_output_size: tuple = args.output_image_size
seed: int = args.seed
models: list = args.model

hyperparameters = {
	#
	"--optimizer": ["adam"],
	"--color-space": ["rgb", "lab"],
	"--learning-rate": [0.0015, 0.0008, 0.0001, 0.0003, 0.002],
	"--example-batch": [512],
	"--example-batch-grid-size": [8],
	# "--use-resnet": [True, False],
	# "--regularization": [0.001, 0.002, 0.003, 0.008, 0.0],
	"--decay-rate": [0.85, 0.90, 0.96],
	"--decay-step": [1000, 10000],
	#"--use-float16": [False, True], #TODO: Cause problem?
	"--loss-fn": ['mse', 'ssim', 'msa'],
	"--seed": [seed],
	"--model": models,
	"--cache-file": [
		"/tmp/super-resolution-cache-" + os.path.basename(os.path.normpath(str(output_dir)))],
	"--epochs": [epochs],
	"--shuffle-data-set-size": [1024],
	"--checkpoint-every-epoch": [0],
	"--data-set-directory": training_dataset_paths,
	"--validation-data-directory": validation_dataset_paths,
	"--batch-size": [batch_size],
}
if test_dataset_paths:
	hyperparameters['--test-set-directory'] = test_dataset_paths

#
keys, values = zip(*hyperparameters.items())
hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#
random.shuffle(hyperparameter_combinations)
print("Number of hyperparameter combinations: " + str(len(hyperparameter_combinations)))

for i, custom_argv in enumerate(hyperparameter_combinations):

	argvlist = []
	for key, value in custom_argv.items():
		argvlist.append(str(key))
		argvlist.append(str(value))

	# Add resolution.
	argvlist.append("--image-size")
	argvlist.append(str(image_size[0]))
	argvlist.append(str(image_size[1]))

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
	if args.use_explicit_cpu:
		argvlist.append("--cpu")

	dcsuperresolution_program(argvlist)
