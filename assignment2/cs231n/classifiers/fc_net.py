from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        hid, cache_1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        out, cache_2 = affine_forward(hid, self.params['W2'], self.params['b2'])
        scores = out
        
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dout = softmax_loss(out, y)
        dhid, grads['W2'], grads['b2'] = affine_backward(dout, cache_2)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dhid, cache_1)
        
        # add the regularization term
        loss = loss + 0.5 * self.reg * ((self.params['W1'] **2).sum() + (self.params['W2'] **2).sum() )
        
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            D = dims[i]
            M = dims[i+1]
            self.params['W%d'%(i+1)] = weight_scale * np.random.randn(D, M)
            self.params['b%d'%(i+1)] = np.zeros(M)

            if (self.normalization=='batchnorm' and i < self.num_layers-1):
                self.params['beta%d'%(i+1)] = np.zeros(M)
                self.params['gamma%d'%(i+1)] = np.ones(M)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        layer = {}
        layer[0] = X
        norm_layer = {}
        af_cache = {}
        re_cache = {}
        bn_cache = {}
        dp_cache = {}
        for i in range(1,self.num_layers):
            af_act, af_cache[i] = affine_forward(layer[i-1], self.params['W%d'%i], self.params['b%d'%i])
            if self.normalization=='batchnorm':
                bn_act, bn_cache[i] =  batchnorm_forward(af_act, self.params['gamma%d'%i], self.params['beta%d'%i], self.bn_params[i-1])
                layer[i], re_cache[i] = relu_forward(bn_act)
            else:
                layer[i], re_cache[i] = relu_forward(af_act)

            if self.use_dropout:
                layer[i], dp_cache[i] = dropout_forward(layer[i], self.dropout_param)
            
        scores, cache_scores = affine_forward(layer[self.num_layers-1], self.params['W%d'%(self.num_layers)], self.params['b%d'%(self.num_layers)])

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dout = softmax_loss(scores, y)

        # Regularization loss
        for i in range(1, self.num_layers+1):
            loss += 0.5 * self.reg * (self.params['W%d'%i] **2).sum()

        dnorm = {}
        dx = {}
        dx[self.num_layers], grads['W%d'%(self.num_layers)], grads['b%d'%(self.num_layers)] = affine_backward(dout, cache_scores)
        grads['W%d'%(self.num_layers)] += self.reg * self.params['W%d'%(self.num_layers)]

        for i in reversed(range(1, self.num_layers)):
            if self.use_dropout:
                dx[i+1] = dropout_backward(dx[i+1], dp_cache[i])
                
            dx_relu = relu_backward(dx[i+1], re_cache[i])
            if self.normalization=='batchnorm':
                dx_bn, grads['gamma%d'%i], grads['beta%d'%i] = batchnorm_backward_alt(dx_relu, bn_cache[i])
                dx[i], grads['W%d'%i], grads['b%d'%i] = affine_backward(dx_bn, af_cache[i])
            else:
                dx[i], grads['W%d'%i], grads['b%d'%i] = affine_backward(dx_relu, af_cache[i])
            grads['W%d'%i] += self.reg * self.params['W%d'%i]

        return loss, grads
