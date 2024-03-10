# !/usr/bin/env python3
import argparse
import logging
import sys
import tensorflow as tf
from util.convert_model import convert_model


def convert_model_program(argv=None):
	parser = argparse.ArgumentParser(
		description='Tensor Lite Converter')

	parser.add_argument('--model-file', dest='model_filepath',
						type=str,
						required=True,
						help='Define filepath to the model')

	parser.add_argument('--output', dest='save_path', type=str, default=None, help='')

	parser.add_argument('--verbosity', type=int, dest='accumulate',
						default=1,
						help='Define the save/load model path')

	args = parser.parse_args(args=argv)

	logger = logging.getLogger('model converter')
	logger.setLevel(logging.INFO)

	with tf.device('/device:CPU:0'):
		generate_model_path = args.model_filepath

		logger.info("Loading Model: {0}".format(generate_model_path))
		model = tf.keras.models.load_model(generate_model_path, compile=False)

		converted_model = convert_model(model=model, dataset=None)
		with open(args.save_path, "wb") as f:
			f.write(converted_model)


# If running the script as main executable
if __name__ == '__main__':
	convert_model_program(sys.argv[1:])
