import numpy as np


def psnr(target, ref):
	# Assume target is RGB/BGR image
	target_data = target.astype(np.float32)
	ref_data = ref.astype(np.float32)

	diff = ref_data - target_data
	diff = diff.flatten('C')

	rmse = np.sqrt(np.mean(diff ** 2.))

	return 20 * np.log10(255. / rmse)
