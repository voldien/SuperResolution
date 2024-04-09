from tensorflow.keras import layers


def create_activation(activation: str, alpha=0.2):
	if activation == "leaky_relu":
		return layers.LeakyReLU(alpha=alpha, dtype='float32')
	elif activation == "relu":
		return layers.ReLU(dtype='float32')
	elif activation == "sigmoid":
		return layers.Activation(activation='sigmoid', dtype='float32')
	else:
		assert "Should never be reached"
