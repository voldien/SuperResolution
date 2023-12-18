import argparse
import itertools
import os
import random
import sys

from superresolution.SuperResolution import dcsuperresolution_program

# Epochs, dataset, batch, size, imagesize, output dir
parser = argparse.ArgumentParser(add_help=True, prog="", description="")

#
parser.add_argument('--epochs', type=int, default=48, dest='epochs',
					help='Set the number of passes that the training set will be trained against.')
#
parser.add_argument('--batch-size', type=int, default=16, dest='batch_size',
					help='number of training element per each batch, during training.')

parser.add_argument('--data-set-directory', type=str, dest='data_sets_directory_paths',
						                 action='append',
					help='Directory path where the images are located dataset images')
#
parser.add_argument('--output-dir', type=str, dest='output_dir',
				default="",
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

args = parser.parse_args(args=sys.argv[1:])


# Extract parameters
epochs :int = args.epochs
output_dir : str = args.output_dir
batch_size : int = args.batch_size
dataset_paths = args.data_sets_directory_paths
image_size :tuple = args.image_size
image_output_size :tuple = args.output_image_size

hyperparameters = {
	#
	"--color-space": ["rgb", "lab"],
	"--learning-rate": [0.0002, 0.0001, 0.0003],
	"--example-batch": [256],
	"--use-resnet": [True, False],
	"--regularization": [0.001, 0.002, 0.003, 0.008, 0.0],
	"--decay-rate": [0.85, 0.90, 0.96],
	"--use-float16": [False, True],
	"--seed": [10350],
	"--model": ['dcsr', 'dscr-post', 'dscr-pre', 'edsr', 'dcsr-ae'],
	"--cache-file": ["/tmp/super-resolution-cache-" + os.path.basename(os.path.normpath(str(output_dir ))) + str(sys.argv[4])],
	"--epochs": [epochs],
	"--shuffle-data-set-size": [2048],
	"--checkpoint-every-epoch": [0],
	"--data-set-directory": [dataset_paths],
	"--batch-size": [args.batch_size],
}

# 
keys, values = zip(*hyperparameters.items())
hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# 
random.shuffle(hyperparameter_combinations)

for i, custom_argv in enumerate(hyperparameter_combinations):

	argvlist = []
	for key, value in custom_argv.items():
		argvlist.append(str(key))
		argvlist.append(str(value))

	# Add resolution.
	argvlist.append("--image-size")
	argvlist.append(str(image_size[0]))
	argvlist.append(str(image_size[1]))

	output_dir = str(os.path.join(sys.argv[5],
								  "It{0}Learning:{1}Reg:{2}Size{3}FP16:{4}DecRate:{5}".format(
									  i,
									  custom_argv["--color-space"],
									  custom_argv["--learning-rate"],
									  custom_argv[
										  "--regularization"],
									  image_size,
									  custom_argv["--use-float16"],
									  custom_argv["--decay-rate"])))
	argvlist.append("--output-dir")
	argvlist.append(output_dir)

	print(argvlist)
	dcsuperresolution_program(argvlist)