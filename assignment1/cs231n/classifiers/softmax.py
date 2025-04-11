from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]  # C
    num_train = X.shape[0]

    for i in range(X.shape[0]):
        # i번째 데이터의 score 계산
        scores = X[i].dot(W)  # Shape: (C,)
        scores -= np.max(scores)

        # loss_i 계산
        probs = np.exp(scores) / np.sum(np.exp(scores))  # Shape: (C, ); 각 클래스에 대한 i번째 데이터의 점수
        loss_i = -np.log(np.exp(scores[y[i]])/ np.sum(np.exp(scores)))
        
        loss += loss_i

        for j in range(num_classes):
            if y[i] == j:
                dW[:, j] += X[i] * (probs[j] - 1)
            else:
                dW[:, j] += X[i] * probs[j]
      
    loss /= num_train  # 전체 loss 평균화
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]  # C
    num_train = X.shape[0]

    scores = X.dot(W)  # Shape: (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)  # numeric stability를 위해 최댓값을 빼기
    
    # Softmax probability 계산
    exp_scores = np.exp(scores)  # Shape: (N, C)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Shape: (N, C)

    correct_class_probs = probs[np.arange(num_train), y]  # Shape: (N,) 
    loss = -np.sum(np.log(correct_class_probs))

    probs[np.arange(num_train), y] -= 1  # 정답 클래스에 대해서는 해당 score에서 -1을 해주어야 함
    dW = X.T.dot(probs)  # X: (N, D), probs: (N, C) => Shape: (D, C)

    # loss, dW 평균화
    loss /= num_train
    dW /= num_train

    # regularization 값 반영
    loss += reg * np.sum(W * W)  # loss에 regularization 항 더하기
    dW += 2 * reg * W  # dW에 reg항 미분값까지 반영

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
