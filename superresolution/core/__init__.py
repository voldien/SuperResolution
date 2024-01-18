import argparse

import tensorflow.keras as keras


class ModelBase(object):
	"""_summary_
	"""

	def load_argument(self) -> argparse.ArgumentParser:
		"""_summary_

		Returns:
			argparse.ArgumentParser: _description_
		"""
		return None

	def create_model(self, input_shape: tuple, output_shape: tuple, **kwargs) -> keras.Model:
		"""_summary_

		Args:
			input_shape (tuple): _description_
			output_shape (tuple): _description_

		Returns:
			keras.Model: _description_
		"""
		return None

	def get_name(self) -> str:
		"""Get Model Name.

		Returns:
			str: _description_
		"""
		return ""

#def validate_shape():
#	return x.shape[1:] == output_shape