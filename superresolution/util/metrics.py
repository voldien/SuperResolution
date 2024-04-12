import tensorflow as tf
import keras
from tensorflow.python.keras.utils import metrics_utils
from util.trainingcallback import compute_normalized_PSNR
import tensorflow_io as tfio


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


@keras.saving.register_keras_serializable(package="superresolution", name="SSIMMetric")
class SSIMMetric(tf.keras.metrics.MeanMetricWrapper):
	@tf.function
	def ssim_loss(self, y_true, y_pred):
		# TODO convert color space.
		y_true_color = None
		y_pred_color = None

		#
		if self.color_space == 'rgb':
			# Remap [-1,1] to [0,1]
			y_true_color = ((y_true + 1.0) * 0.5)
			y_pred_color = ((y_pred + 1.0) * 0.5)
		elif self.color_space == 'lab':
			# Remap [-1,1] -> [-128, 128] -> [0,1]
			y_true_color = tfio.experimental.color.lab_to_rgb(y_true * 128)
			y_pred_color = tfio.experimental.color.lab_to_rgb(y_pred * 128)
		else:
			assert 0

		return tf.reduce_mean(tf.image.ssim(y_true_color, y_pred_color, max_val=1.0, filter_size=11,
											filter_sigma=1.5, k1=0.01, k2=0.03))

	def __init__(self, name: str = "SSIM", color_space: str = "rgb", dtype=None):
		def ssim(y_true, y_pred):
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
			return tf.cast(self.ssim_loss(y_true, y_pred), tf.float32)
		super().__init__(ssim, name, dtype=dtype)
		self.color_space = color_space
