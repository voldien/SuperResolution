from util.trainingcallback import compute_normalized_PSNR
from keras.src.applications import VGG19
from keras.src.applications.vgg16 import preprocess_input
from keras.src.losses import LossFunctionWrapper
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
import keras
import tensorflow.python.keras.backend as K
import tensorflow.keras as keras
import tensorflow_io as tfio

@keras.saving.register_keras_serializable(package="superresolution", name="VGG16Error")
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

		# Remap [-1,1] -> [0,1] | [RGB] -> [BGR]
		h1_list = self.model(preprocess_input((y_pred + 1) * 0.5))
		h2_list = self.model(preprocess_input((y_true + 1) * 0.5))

		rc_loss: float = 0.0
		for h1, h2, weight in zip(h1_list, h2_list, self.selected_layer_weights):
			h1 = K.batch_flatten(h1)
			h2 = K.batch_flatten(h2)

			rc_loss = rc_loss + weight * K.sum(K.square(h1 - h2), axis=-1)

		return rc_loss

	def __init__(
		self,
		reduction=losses_utils.ReductionV2.SUM,
		name="mean_absolute_error",
	):
		super().__init__(self.vgg16_loss, name=name, reduction=reduction)
		self.VGG = None
		self.model = None



@keras.saving.register_keras_serializable(package="superresolution", name="loss_ssim")
class SSIMError(LossFunctionWrapper):

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

		return (1 - tf.reduce_mean(tf.image.ssim(y_true_color, y_pred_color, max_val=1.0, filter_size=11,
												 filter_sigma=1.5, k1=0.01, k2=0.03)))


	def __init__(
		self,
		reduction=losses_utils.ReductionV2.SUM,
		name: str="ssim_loss_function",
		color_space : str = None,
	):
		super().__init__(self.ssim_loss, name=name, reduction=reduction)
		self.VGG = None
		self.model = None
		self.color_space = color_space


def psnr_loss(y_true, y_pred):  # TODO: fix equation.
	return 20.0 - compute_normalized_PSNR(y_true, y_pred)

def total_variation_loss(y_true, y_pred):  # TODO: fix equation.
	return 1.0 - tf.reduce_sum(tf.image.total_variation(y_true, y_pred))