import numpy as np

def decode_ecoc_hamming(code_words, predictions):
	# ecoc hamming decoding method
	num_classes = code_words.shape[0]
	distances = np.zeros((len(predictions), num_classes))
	
	for i in range(num_classes):
		mask = code_words[i] != 0
		distances[:, i] = np.sum(predictions[:, mask] != code_words[i, mask], axis=1)
	
	return np.argmin(distances, axis=1)

def decode_ecoc_euclidean(code_words, predictions):
	# ecoc euclidean decoding method
	num_classes = code_words.shape[0]
	distances = np.zeros((len(predictions), num_classes))
	
	for i in range(num_classes):
		mask = code_words[i] != 0
		diff = predictions[:, mask] - code_words[i, mask]
		distances[:, i] = np.sqrt(np.sum(diff**2, axis=1))
	
	return np.argmin(distances, axis=1)

def decode_ecoc_prob_loss_based(code_words, probabilities):
	# ecoc probability decoding method
	num_classes = code_words.shape[0]
	losses = np.zeros((len(probabilities), num_classes))
	
	for i in range(num_classes):
		mask = code_words[i] != 0
		losses[:, i] = np.sum(np.where(code_words[i, mask] == 1, 
									1 - probabilities[:, mask],
									probabilities[:, mask]), axis=1)
	
	return np.argmin(losses, axis=1)

def decode_ecoc_loss_based(code_words, decision_functions):
	# ecoc loss_based decoding method
	num_classes = code_words.shape[0]
	losses = np.zeros((len(decision_functions), num_classes))
	
	for i in range(num_classes):
		mask = code_words[i] != 0
		z = decision_functions[:, mask] * code_words[i, mask]
		loss = 1 - z  # This is equivalent to max(1 - z, 0) when z <= 1
		losses[:, i] = np.sum(np.where(loss > 0, loss, 0), axis=1)

	return np.argmin(losses, axis=1)

def decode_ecoc(code_words, data, method='loss_based'):
    """
    Entry function for ECOC decoding.
    
    Parameters:
    code_words : numpy.ndarray
        The ECOC matrix.
    data : numpy.ndarray
        The data to be decoded. This could be predictions, probabilities, or decision function values,
        depending on the decoding method.
    method : str, optional
        The decoding method to use. Options are 'hamming', 'euclidean', 'prob_loss_based', or 'loss_based'.
        Default is 'loss_based'.
    
    Returns:
    numpy.ndarray
        The predicted class labels.
    """
    if method == 'hamming':
        return decode_ecoc_hamming(code_words, data)
    elif method == 'euclidean':
        return decode_ecoc_euclidean(code_words, data)
    elif method == 'prob_loss_based':
        return decode_ecoc_prob_loss_based(code_words, data)
    elif method == 'loss_based':
        return decode_ecoc_loss_based(code_words, data)
    else:
        raise ValueError("Unsupported decoding method. Use 'hamming', 'euclidean', 'prob_loss_based', or 'loss_based'.")