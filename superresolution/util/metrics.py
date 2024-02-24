import tensorflow as tf
from keras.src.applications import VGG19
from keras.src.applications.vgg16 import preprocess_input
from keras.src.losses import LossFunctionWrapper
import keras
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.utils import metrics_utils, losses_utils
from util.trainingcallback import compute_normalized_PSNR


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


class VGG16Error(LossFunctionWrapper):

	selected_layers = ['block1_conv1', 'block2_conv2',
									   "block3_conv3", 'block4_conv3', 'block5_conv4']
	selected_layer_weights = [1.0, 4.0, 4.0, 8.0, 16.0]

	@tf.function
	def vgg16_loss(self, y_true, y_pred):
		# Construct model if not loaded.
		if self.VGG is None:
			self.VGG = VGG19(weights='imagenet')
			outputs = [self.VGG.get_layer(
				l).output for l in self.selected_layers]
			self.model = keras.Model(self.VGG.input, outputs)

		# Resize to fit the model.
		y_pred = tf.image.resize(y_pred, (224, 224))
		y_true = tf.image.resize(y_true, (224, 224))

		# Remap [-1,1] -> [0,1]
		h1_list = self.model((y_pred + 1)*0.5)
		h2_list = self.model((y_true + 1)*0.5)

		rc_loss: float = 0.0
		for h1, h2, weight in zip(h1_list, h2_list, self.selected_layer_weights):

			h1 = K.batch_flatten(h1)
			h2 = K.batch_flatten(h2)

			rc_loss = rc_loss + weight * K.sum(K.square(h1 - h2), axis=-1)

		return rc_loss

	def __init__(
		self,
		reduction=losses_utils.ReductionV2.AUTO,
		name="mean_absolute_error",
	):
		super().__init__(self.vgg16_loss, name=name, reduction=reduction)
		self.VGG = None
		self.model = None
