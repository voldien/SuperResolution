import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import asarray
from skimage.color import lab2rgb


def generate_image(model, latent_space):
	return model(latent_space, training=False)


def showResult(model, image_batch_dataset, color_space, maxNumImages=6):

	batch_iter = iter(image_batch_dataset)

	data_image_batch, expected_image_batch = next(batch_iter)

	output = model.predict(data_image_batch, verbose=0)

	nrElements = min(len(output), maxNumImages)

	rows = 3 + 3
	fig = plt.figure(figsize=(maxNumImages * 2, 5 * 2))
	for i in range(nrElements):

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

		plt.subplot(rows, maxNumImages, maxNumImages * 0 + i + 1)
		plt.imshow((asarray(data_image).astype(dtype='float32')))
		plt.axis("off")

		plt.subplot(rows, maxNumImages, maxNumImages *1 + i + 1)
		plt.imshow((asarray(expected_image).astype(dtype='float32')))
		plt.axis("off")

		result_image = None
		# Convert color-space to normalize coordinates [0,1]
		if color_space == 'rgb':
			# Convert the raw encoded image to RGB color space.
			result_image = (output[i % len(data_image_batch)] + 1.0) * 0.5
		elif color_space == 'lab':
			# Convert the raw encoded image to LAB color space.
			result_image = lab2rgb(output[i % len(data_image_batch)] * 128)

		plt.subplot(rows, maxNumImages, maxNumImages * 2 + i + 1)
		plt.imshow(result_image[:, :, 0], cmap='gray')
		plt.axis("off")

		plt.subplot(rows, maxNumImages, maxNumImages * 3 + i + 1)
		plt.imshow(result_image[:, :, 1], cmap='Blues')
		plt.axis("off")

		plt.subplot(rows, maxNumImages, maxNumImages * 4 + i + 1)
		plt.imshow(result_image[:, :, 2], cmap='Greens')
		plt.axis("off")

		plt.subplot(rows, maxNumImages, maxNumImages * 5 + 1 + i)
		plt.imshow(asarray(result_image).astype(dtype='float32'))
		plt.axis("off")

		if len(data_image_batch) - 1 == i:
			data_image_batch, expected_image_batch = next(batch_iter)
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
