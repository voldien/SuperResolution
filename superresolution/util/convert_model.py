import tensorflow as tf

def convert_model(model, dataset=None):

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.inference_input_type = tf.float32
	converter.inference_output_type = tf.float32
	converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
							# enable TensorFlow Lite ops.
							tf.lite.OpsSet.TFLITE_BUILTINS,
							# enable TensorFlow ops.
							tf.lite.OpsSet.SELECT_TF_OPS
							]

	converter.post_training_quantize = True
	
	if dataset:
		converter.representative_dataset = tf.lite.RepresentativeDataset(
			dataset)

	tflite_model = converter.convert()

	return tflite_model
