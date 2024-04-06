import tensorflow as tf
import keras
from tensorflow.python.keras.utils import metrics_utils
from util.trainingcallback import compute_normalized_PSNR


@keras.saving.register_keras_serializable(package="superresolution", name="PSNRMetric")
class PSNRMetric(tf.keras.metrics.MeanMetricWrapper):
	def __init__(self, name="PSNR", dtype=None):
		def psnr(y_true, y_pred):
			[
				y_pred,
				y_true,
			], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
				[y_pred, y_true]
			)
			y_true.shape.assert_is_compatible_with(y_pred.shape)
			if y_true.dtype != y_pred.dtype:
				y_pred = tf.cast(y_pred, y_true.dtype)

			# TODO determine cast type.
			return tf.cast(compute_normalized_PSNR(y_true, y_pred), tf.float32) * 0.01

		super().__init__(psnr, name, dtype=dtype)

