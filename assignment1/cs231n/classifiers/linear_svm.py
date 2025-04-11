from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # X[i] = (1 x 3073), W = 3073x10
        correct_class_score = scores[y[i]]  # y[i] = X[i]의 정답, scores[y[i]] = X[i]의 정답 class에 대한 점수
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1  (hinge loss의 마진 값 = 1)
            if margin > 0:  # loss에 기여하는 경우
                loss += margin
                # W_s_j에 대해 loss를 미분한 값 = X[i]이고, W_s_y[i]에 대해 loss를 미분한 값 = -X[i]..?
                dW[:, j] += X[i]  # 안 좋은 점수를 낸 클래스에 대해 gradient 증가 -> update시 ... ?
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train  # loss도 평균화했으니 dW도 일관성을 위해서???

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)  # R(W) = \sigma_k \sigma_l W_{k,l}^2 -> 이만큼 penalzing with regularization..
    dW += 2 * reg * W  # dL/dW를 구해야 하는데, 지금까지는 data loss에 대한 미분값만 구함 -> 마지막 정규화 항에 대해서도 구해야됨

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #### 각 class별 유사도 = scores 계산
    scores = X.dot(W)  # (N, C); N은 데이터 개수, C는 클래스 개수

    #### 각 이미지에 대한 정답 class 점수 array
    correct_class_scores = scores[np.arange(scores.shape[0]), y]  # (N,), np.arange()와 y에 대해 0부터 증가하면서 배열에 접근하는 방법
    
    #### margin 값들 계산
    # np.newaxis는 새로운 축을 하나 더 만들어서 차원을 하나 늘리는 것
    # scores는 (N, C)이고, correct_class_scores 는 (N, )이라 차원이 맞지 않음 -> 뺄셈 불가능 
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)   # Shape: (N, C)
    margins[np.arange(margins.shape[0]), y] = 0  # 정답 class score에 대해서는 loss 값을 0으로 (고려 안함)

    #### loss 값 계산
    loss = np.sum(margins) / X.shape[0]  # N(데이터 개수) 평균화
    loss += reg * np.sum(W * W)  # 정규화 항 더하기

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    margin_mask = (margins > 0).astype(float)  # margin > 0인지 여부의 True/False를 float로 변환한 배열, shape: (N, C)

    # margin_mask의 각 행을 모두 더함 = 각 데이터에 대해 margin > 0 인 클래스들의 개수를 구함
    # -> 각 데이터의 정답 클래스에 해당하는 margin_mask 값에서, 그 데이터에 대해 margin > 0 인 클래스의 개수를 뺌
    # => 각 데이터의 정답 클래스의 loss값에 대한 기여도 조정......?
    margin_mask[np.arange(margin_mask.shape[0]), y] -= np.sum(margin_mask, axis = 1) 
    dW = X.T.dot(margin_mask) / X.shape[0]  # X의 transpose로 곱셈 = 차원을 맞추기 위함 -> 곱셈 결과가 달라지지 않나??
    dW += 2 * reg * W  # Add regularization gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
