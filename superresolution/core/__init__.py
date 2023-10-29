import argparse

import tensorflow.keras as keras


class ModelBase(object):
	def load_argument(self) -> argparse.ArgumentParser:
		"""Load in the file for extracting text."""
		return None

	def create_model(self, input_shape: tuple, output_shape: tuple, **kwargs) -> keras.Model:
		"""Extract text from the currently loaded file."""
		return None

	def get_name(self) -> str:
		""""""
		return ""
