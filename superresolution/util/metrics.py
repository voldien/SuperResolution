import tensorflow as tf
from keras.metrics import base_metric
from util.trainingcallback import compute_PSNR


class PSNRMetric(base_metric.MeanMetricWrapper):
	def __init__(self, name="pbnr", dtype=None):
		def psnr(y_true, y_pred):
			return tf.cast(compute_PSNR(y_true, y_pred), tf.backend.floatx())

		super().__init__(psnr, name, dtype=dtype)
