import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #  We want to move other class weights away from this vector
        dW[:,j] += X[i]
        #  We want to move the correct class weights towards this vector
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  # Gradient should be an average, as well have regularization added to it
  dW /= num_train
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  num_train = X.shape[0]
  
  # Calculate scores & identify the correct class scores for each sample
  scores = X.dot(W)
  correct_scores = scores[np.arange(num_train),y]
  
  margins = np.maximum(0, scores - correct_scores[:, np.newaxis] + 1.0) # note delta = 1
  margins[np.arange(num_train),y] = 0 # negate margin from correct class (margin in this case is 0 + delta)

  # Sum up the loss, average it, and regularize
  loss = np.sum(margins)
  loss /= num_train
  loss += reg * np.sum(W * W)

  # Flag all instances of margins being too small
  X_mask = np.zeros(margins.shape)
  X_mask[margins > 0] = 1

  # Count instances of small margins to reduce the correct class weight by
  count = np.sum(X_mask,1)
  X_mask[np.arange(num_train),y] = -count
  
  # Calculate gradient, as well as obtain its average & add regularization
  dW += (X.T).dot(X_mask)
  dW /= num_train
  dW += reg*W

  return loss, dW
