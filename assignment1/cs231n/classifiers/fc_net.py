from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


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

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
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

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 첫번째 레이어; input 입력, hidden 출력
        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros_like([hidden_dim, ])

        # 두번째 레이어; hidden 입력, num_classes 출력
        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros_like([num_classes, ])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

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
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        layer1_output, layer1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        relu_output, relu_cache = relu_forward(layer1_output)
        layer2_output, layer2_cache = affine_forward(relu_output, self.params['W2'], self.params['b2'])
        
        scores = layer2_output

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # scores -= np.max(scores, axis=1, keepdims=True)

        # loss, dloss = softmax_loss(scores, y)
        # loss += 0.5 * self.reg * (np.sum(self.params['W1'] **2) + np.sum(self.params['W2']**2))

        # dx2, dw2, db2 = affine_backward(dloss, layer2_cache)
        # dw1, dw1, db1 = affine_relu_backward(dx2, (layer1_cache, relu_cache))
        
        # dw1 += self.reg * self.params['W1'] 
        # dw2 += self.reg * self.params['W2']

        # grads = {
        #   'W1': dw1,
        #   'b1': db1,
        #   'W2': dw2,
        #   'b2': db2
        # }

        loss, softmax_dx = softmax_loss(scores, y)

        # 뒤로 계속 가기!!

        layer2_dx, layer2_dw, layer2_db = affine_backward(softmax_dx, layer2_cache)
        layer2_dw += self.reg * self.params['W2']  # 정규화항 미분 -> 2를 왜 곱하면 안되는지......????
        grads['W2'] = layer2_dw
        grads['b2'] = layer2_db
        layer2_reg = 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])

        relu_dx = relu_backward(layer2_dx, relu_cache)

        layer1_dx, layer1_dw, layer1_db = affine_backward(relu_dx, layer1_cache)
        layer1_dw += self.reg * self.params['W1']  # 여기도...
        grads['W1'] = layer1_dw
        grads['b1'] = layer1_db
        layer1_reg = 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])

        # loss에 각 W의 정규화항을 더해서 최종 loss 값 구하기
        loss += layer2_reg + layer1_reg

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
