import matplotlib.pyplot as plt
from util.util import convert_nontensor_color_space
import tensorflow as tf
from numpy import asarray
from tensorflow.python.data import Dataset


def show_expect_predicted_result(model: tf.keras.Model, image_batch_dataset: Dataset, color_space: str, nr_col=6):
	batch_iter = iter(image_batch_dataset)
	data_image_batch, expected_image_batch = batch_iter.next()

	output = model.predict(data_image_batch, verbose=0)

	rows = 3 + 3

	fig = plt.figure(figsize=(nr_col * 2, 5 * 2))
	for i in range(nr_col):

		data_image = convert_nontensor_color_space(data_image_batch[i % len(data_image_batch)], color_space=color_space)
		expected_image = convert_nontensor_color_space(expected_image_batch[i % len(expected_image_batch)],
													   color_space=color_space)

		# Display Input Training Data.
		plt.subplot(rows, nr_col, nr_col * 0 + i + 1)
		plt.imshow((asarray(data_image).astype(dtype='float32')))
		plt.axis("off")

		# Display Expected(Output) Train Data.
		plt.subplot(rows, nr_col, nr_col * 1 + i + 1)
		plt.imshow((asarray(expected_image).astype(dtype='float32')))
		plt.axis("off")

		# Convert color-space to normalize coordinates [0,1]
		output_result_raw = output[i % len(data_image_batch)]
		result_image_rgb_encoding = convert_nontensor_color_space(output[i % len(data_image_batch)],
																  color_space=color_space)

		plt.subplot(rows, nr_col, nr_col * 2 + i + 1)
		if color_space == 'lab':
			plt.imshow(output_result_raw[:, :, 0], cmap='gray')
		elif color_space == 'rgb':
			plt.imshow(output_result_raw[:, :, 0], cmap='Reds')
		plt.axis("off")

		plt.subplot(rows, nr_col, nr_col * 3 + i + 1)
		plt.imshow(output_result_raw[:, :, 1], cmap='Blues')
		plt.axis("off")

		plt.subplot(rows, nr_col, nr_col * 4 + i + 1)
		plt.imshow(output_result_raw[:, :, 2], cmap='Greens')
		plt.axis("off")

		#
		plt.subplot(rows, nr_col, nr_col * 5 + 1 + i)
		plt.imshow(asarray(result_image_rgb_encoding).astype(dtype='float32'))
		plt.axis("off")

		if len(data_image_batch) - 1 == i:
			data_image_batch, expected_image_batch = batch_iter.next()
			output = model.predict(data_image_batch, verbose=0)

	fig.subplots_adjust(wspace=0.05, hspace=0.05)
	plt.close()
	return fig
