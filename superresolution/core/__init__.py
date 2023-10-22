import tensorflow.keras as keras
import argparse

class ModelBase(object):
	def load_argument(self) -> argparse.ArgumentParser:
		"""Load in the file for extracting text."""
		pass

	def create_model(self, input_shape, output_shape, **kwargs) -> keras.Model:
		"""Extract text from the currently loaded file."""
		pass

	def get_name(self) -> str:
		return ""