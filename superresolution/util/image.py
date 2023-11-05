import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import asarray
from skimage.color import lab2rgb


def showResult(model: tf.keras.Model, image_batch_dataset, color_space: str, nrCol=6):
	batch_iter = iter(image_batch_dataset)
	data_image_batch, expected_image_batch = batch_iter.next()

	output = model.predict(data_image_batch, verbose=0)

	rows = 3 + 3

	fig = plt.figure(figsize=(nrCol * 2, 5 * 2))
	for i in range(nrCol):

		data_image = None
		expected_image = None

		# Convert color-space to normalize coordinates [0,1]
		if color_space == 'rgb':
			# Convert the raw encoded image to RGB color space.
			data_image = (data_image_batch[i % len(data_image_batch)] + 1.0) * 0.5
			expected_image = (expected_image_batch[i % len(expected_image_batch)] + 1.0) * 0.5
		elif color_space == 'lab':
			# Convert the raw encoded image to LAB color space.
			data_image = lab2rgb(data_image_batch[i % len(data_image_batch)] * 128)
			expected_image = lab2rgb(expected_image_batch[i % len(expected_image_batch)] * 128)

		# Display Input Training Data.
		plt.subplot(rows, nrCol, nrCol * 0 + i + 1)
		plt.imshow((asarray(data_image).astype(dtype='float32')))
		plt.axis("off")

		# Display Expected(Output) Train Data.
		plt.subplot(rows, nrCol, nrCol * 1 + i + 1)
		plt.imshow((asarray(expected_image).astype(dtype='float32')))
		plt.axis("off")

		result_image_rgb_encoding = None
		# Convert color-space to normalize coordinates [0,1]
		output_result = output[i % len(data_image_batch)]
		if color_space == 'rgb':
			# Convert the raw encoded image to RGB color space.
			result_image_rgb_encoding = asarray((output[i % len(data_image_batch)] + 1.0) * 0.5).astype(dtype='float32')
		elif color_space == 'lab':
			# Convert the raw encoded image to LAB color space.
			result_image_rgb_encoding = asarray(lab2rgb(output[i % len(data_image_batch)] * 128)).astype(
				dtype='float32')

		plt.subplot(rows, nrCol, nrCol * 2 + i + 1)
		plt.imshow(output_result[:, :, 0], cmap='gray')
		plt.axis("off")

		plt.subplot(rows, nrCol, nrCol * 3 + i + 1)
		plt.imshow(output_result[:, :, 1], cmap='Blues')
		plt.axis("off")

		plt.subplot(rows, nrCol, nrCol * 4 + i + 1)
		plt.imshow(output_result[:, :, 2], cmap='Greens')
		plt.axis("off")

		#
		plt.subplot(rows, nrCol, nrCol * 5 + 1 + i)
		plt.imshow(asarray(result_image_rgb_encoding).astype(dtype='float32'))
		plt.axis("off")

		if len(data_image_batch) - 1 == i:
			data_image_batch, expected_image_batch = batch_iter.next()
			output = model.predict(data_image_batch, verbose=0)

	fig.subplots_adjust(wspace=0.05, hspace=0.05)
	plt.close()
	return fig


def generate_grid_image(model, latent_spaces, color_space, figsize=(8, 8), subplotsize=(3, 3)):
	latent_space_c = len(latent_spaces[0])

	fig = plt.figure(figsize=figsize)
	for i in range(subplotsize[0] * subplotsize[1]):

		#
		latent = latent_spaces[i]

		# Raw Generated image.
		generated_images = model(tf.reshape(
			latent, [1, latent_space_c]), training=False)

		# Convert color-space to normalize coordinates [0,1]
		if color_space == 'rgb':
			# Convert the raw encoded image to RGB color space.
			generated_images = (generated_images + 1) * 0.5
		elif color_space == 'lab':
			# Convert the raw encoded image to LAB color space.
			generated_images = lab2rgb(generated_images * 128)

		# Select active subplot element.
		plt.subplot(subplotsize[0], subplotsize[1], i + 1)

		rgb_image = generated_images[0, :, :, 0:3]

		# Present image.
		plt.imshow(asarray(rgb_image).astype('float32'))
		plt.axis('off')

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.close()
	return fig
