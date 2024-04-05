from tensorflow.keras import layers


def create_activation(activation: str):
	if activation == "leaky_relu":
		return layers.LeakyReLU(alpha=0.2, dtype='float32')
	elif activation == "relu":
		return layers.ReLU(dtype='float32')
	elif activation == "sigmoid":
		return layers.Activation(activation='sigmoid', dtype='float32')
	else:
		assert "Should never be reached"
