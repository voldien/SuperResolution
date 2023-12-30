import tensorflow as tf
from keras.dtensor import utils as dtensor_utils
from keras.metrics import base_metric
from keras.utils import metrics_utils
from util.trainingcallback import compute_normalized_PSNR


class PSNRMetric(base_metric.MeanMetricWrapper):
	@dtensor_utils.inject_mesh
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

			return tf.cast(compute_normalized_PSNR(y_true, y_pred), tf.float32)  # TODO determine cast type.

		super().__init__(psnr, name, dtype=dtype)
