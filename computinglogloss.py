import numpy as np
def compute_log_loss(predicted, actual, eps=1e-14):
 """ Computes the logarithmic loss between predicted and
 actual when these are 1D arrays.

 :param predicted: The predicted probabilities as floats between 0-1
 :param actual: The actual binary labels. Either 0 or 1.
 :param eps (optional): log(0) is inf, so we need to offset our
 predicted values slightly by eps from 0 or 1.
 """
 predicted = np.clip(predicted, eps, 1 - eps)
 loss = -1 * np.mean(actual * np.log(predicted)
 + (1 - actual)
 * np.log(1 - predicted))

 return loss

def main():
	actuals = np.array([ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.])
	correct_confident = np.array([ 0.95,  0.95,  0.95,  0.95,  0.95,  0.05,  0.05,  0.05,  0.05,  0.05])
	correct_not_confident = np.array([ 0.65,  0.65,  0.65,  0.65,  0.65,  0.35,  0.35,  0.35,  0.35,  0.35])
	wrong_not_confident = np.array([ 0.35,  0.35,  0.35,  0.35,  0.35,  0.65,  0.65,  0.65,  0.65,  0.65])
	wrong_confident = np.array([ 0.05,  0.05,  0.05,  0.05,  0.05,  0.95,  0.95,  0.95,  0.95,  0.95])
	actual_labels = np.array([ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.])

	# Compute and print log loss for 1st case
	correct_confident = compute_log_loss(correct_confident,actual_labels)
	print("Log loss, correct and confident: {}".format(correct_confident)) 

	# Compute log loss for 2nd case
	correct_not_confident = compute_log_loss(correct_not_confident, actual_labels)
	print("Log loss, correct and not confident: {}".format(correct_not_confident)) 

	# Compute and print log loss for 3rd case
	wrong_not_confident = compute_log_loss(wrong_not_confident, actual_labels)
	print("Log loss, wrong and not confident: {}".format(wrong_not_confident)) 

	# Compute and print log loss for 4th case
	wrong_confident = compute_log_loss(wrong_confident,actual_labels)
	print("Log loss, wrong and confident: {}".format(wrong_confident)) 

	# Compute and print log loss for actual labels
	actual_labels = compute_log_loss(actual_labels, actual_labels)
	print("Log loss, actual labels: {}".format(actual_labels)) 



if __name__ == '__main__':
	main()