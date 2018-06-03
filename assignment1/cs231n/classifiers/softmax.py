import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    # shift scores so that largest value is 0 for stability
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]

    # add to loss the log of the exponential of all scoures
    dscores = np.exp(scores)
    loss -= np.log( np.exp(correct_class_score) / np.sum(dscores) )

    # compute gradient scores, while placing emphasis on correct score
    dscores /= np.sum(dscores)
    dscores[y[i]] -= 1
    for j in range(num_classes):
      dW[:,j] += dscores[j]*X[i,:]
    
  # Average loss & gradient
  loss /= num_train
  dW /= num_train

  # Regularization
  loss += reg * np.sum(W * W)
  dW += reg * W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  # number stability
  scores -= np.amax(scores, 1)[:, np.newaxis]
  correct_class_score = scores[np.arange(num_train), y]

  # compute loss
  dscores = np.exp(scores)
  loss -= np.sum(np.log(np.exp(correct_class_score) / np.sum(dscores, 1)))

  # compute
  dscores /= np.sum(dscores, 1)[:, np.newaxis]
  dscores[np.arange(num_train), y] -= 1
  dW += (X.T).dot(dscores)

  # Average loss & gradient
  loss /= num_train
  dW /= num_train

  # Regularization
  loss += reg * np.sum(W * W)
  dW += reg * W

  return loss, dW

