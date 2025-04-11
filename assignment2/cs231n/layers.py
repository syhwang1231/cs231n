from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]  # N (입력 샘플 개수)
    reshaped_x = x.reshape(num_train, -1)  # Shape: (N, D)
    out = np.dot(reshaped_x, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]  # N
    
    # 1. dx 계산 - dx에 대한 local gradient = w (곱했기 때문에)
    dx_flat = dout.dot(w.T)  # Shape: (N, D)
    dx = dx_flat.reshape(x.shape)  # Shape: (N, d1, ... d_k)

    # 2. dw 계산 - local gradient with respect to w = x (곱했기 때문에)
    dw = x.reshape(num_train, -1).T.dot(dout)  # (D, N) x (N, M) -> (D, M)

    # 3. db 계산 - local gradient with respect to b = dL/dy = upstream gradient와 동일함
    db = np.sum(dout, axis=0)  # (N, M) -> (M,)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    zero_mask = x > 0  # x > 0 결과에 대한 t/f (1/0) 정보를 담은 matrix; Shape: same as x
    out = x * zero_mask  # x와 zero_mask를 곱하면 0보다 큰 값들만 살아남음

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ReLU에서는 0보다 큰 x(input)에 대해서만 upstream gradient값을 보내주면 됨
    zero_mask = cache > 0
    dx = zero_mask * dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0] # N

    x -= x.max(axis=1, keepdims=True)  # numeric stability를 위해 최댓값을 빼기
    
    exp_scores = np.exp(x)  # Shape: (N, C)
    
    # 정답 class의 softmax probability
    correct_class_exp_scores = np.exp(x[np.arange(num_train), y])  # Shape: (N,) 
    scores_exp_sum = np.sum(exp_scores, axis=1)
    
    loss = - np.sum(np.log(correct_class_exp_scores / scores_exp_sum))
    loss /= num_train  # normalize

    # gradient 계산
    dx = np.zeros_like(x)
    dx += np.exp(x) / scores_exp_sum[:, np.newaxis]  # x_{i, j}의 softmax의 결과를 저장
    dx[np.arange(num_train), y] -= 1  # 정답 클래스에 대해서는 해당 score에서 -1을 해주어야 함; Shape: (N, C)
    dx /= num_train  # normalize

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 각 dimension에 대해 mean, variance 계산
        mean = np.mean(x, axis = 0)  # Shape: (D,)
        var = np.var(x, axis = 0)

        normalized_x = (x - mean) / np.sqrt(var + eps)
        out = gamma * normalized_x + beta
        cache = (x, mean, var, normalized_x, gamma, beta, eps)  # backward때 어떤게 필요할지 모르겠는데..;;

        # update running mean, running variance
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        normalized_x = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * normalized_x + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, mean, var, normalized_x, gamma, beta, eps = cache

    N = dout.shape[0]
    D = dout.shape[1]

    # dgamma = dout의 1부터 N까지 합 * x_hat
    dgamma = np.sum(dout * normalized_x, axis = 0)  # (D, )

    # dbeta = dout의 1부터 N까지 합
    dbeta = np.sum(dout, axis = 0)  # (D, )

    dnormalized_x = gamma * dout  # (N, D)
    
    dvar = np.sum(-0.5 * (var + eps) ** -1.5 * (x - mean) * dnormalized_x, axis = 0)  # (D,)
    dx_minus_mean = 2 * (x - mean) * np.ones((N, D)) * dvar / N + 1 / np.sqrt(var + eps) * dnormalized_x 
    dmean = -1 * np.sum(dx_minus_mean, axis = 0)  # (D,)
    
    dx = np.ones((N, D)) * dmean / N + dx_minus_mean  # (N, D)   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, mean, var, normalized_x, gamma, beta, eps = cache

    N = dout.shape[0]
    D = dout.shape[1]

    dgamma = np.sum(dout * normalized_x, axis = 0)
    dbeta = np.sum(dout, axis = 0)

    dx_hat = dout * gamma  # (N, D)

    dvar = np.sum(dx_hat * -0.5 * (x - mean) * (var + eps) ** (-1.5), axis = 0)  # dl/dvar 의미 = upstream gradient(=dx_hat) * local gradient

    dmean = np.sum(dx_hat / -np.sqrt(var + eps) + dvar * (x - mean) * -2 / N, axis = 0)  # dl/dmean = dl/dx_hat * dx_hat/dmean + dl/dv * dv/dmean

    dx = dx_hat / np.sqrt(var + eps) + dvar * (x - mean) * 2 / N + dmean / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 각 data의 feature들에 대해 mean, variance 계산
    feature_mean = np.mean(x, axis = 1)[:, np.newaxis]  # Shape: (N,) -> (N, 1)  (N,D)와 차원을 맞추기 위함
    feature_var = np.var(x, axis = 1)[:, np.newaxis]  # Shape: (N,) -> (N, 1)

    normalized_x = (x - feature_mean) / np.sqrt(feature_var + eps)  # Shape: (N, D)
    out = gamma * normalized_x + beta
    cache = (x, feature_mean, feature_var, normalized_x, gamma, beta, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, feature_mean, feature_var, normalized_x, gamma, beta, eps = cache

    N = dout.shape[0]
    D = dout.shape[1]

    dgamma = np.sum(dout * normalized_x, axis = 0) # (D,)
    dbeta = np.sum(dout, axis = 0)  # (D,)

    dnormalized_feature = dout * gamma  # (N, D)

    # Shape: (N, 1)
    dvar = np.sum(dnormalized_feature * -0.5 * (x - feature_mean) * (feature_var + eps) ** (-1.5), axis = 1)[:, np.newaxis]  # dl/dvar 의미 = upstream gradient(=dx_hat) * local gradient

    # Shape: (N, 1)
    dmean = np.sum(dnormalized_feature / -np.sqrt(feature_var + eps) + dvar * (x - feature_mean) * -2 / D, axis = 1)[:, np.newaxis]  # dl/dmean = dl/dx_hat * dx_hat/dmean + dl/dv * dv/dmean

    # Shape: (N, D)
    dx = dnormalized_feature / np.sqrt(feature_var + eps) + dvar * (x - feature_mean) * 2 / D + dmean / D

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p  # inverted dropout
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    H = x.shape[2]
    W = x.shape[3]

    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]

    stride = conv_param['stride']
    pad = conv_param['pad']

    output_H = 1 + ((H + pad * 2) - HH) // stride
    output_W = 1 + ((W + pad * 2) - WW) // stride

    x_pad = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values = 0)

    out = np.zeros((N, F, output_H, output_W))

    for n in range(N):
        for f in range(F):
            for new_h in range(output_H):
                for new_w in range(output_W):
                    h_start = new_h * stride
                    w_start = new_w * stride
                    out[n][f][new_h][new_w] = np.sum(w[f, :, :, :] * x_pad[n, :, h_start:h_start+HH, w_start:w_start+WW]) + b[f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache

    N = x.shape[0]
    H = x.shape[2]
    W = x.shape[3]

    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]

    stride = conv_param['stride']
    pad = conv_param['pad']

    output_H = 1 + ((H + pad * 2) - HH) // stride
    output_W = 1 + ((W + pad * 2) - WW) // stride

    x_pad = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values = 0)

    # dw, dw, db shape 세팅
    dx = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for new_h in range(output_H):
                for new_w in range(output_W):
                    h_start = new_h * stride
                    w_start = new_w * stride
                    # db = dout
                    db[f] += dout[n, f, new_h, new_w]
                    
                    # dx = dout * w
                    dx[n, :, h_start:h_start+HH, w_start:w_start+WW] += dout[n, f, new_h, new_w] * w[f, :, :, :]
                    
                    # dw = dout * x
                    dw[f, :, :, :] += dout[n, f, new_h, new_w] * x_pad[n, :, h_start:h_start+HH, w_start:w_start+WW]
    
    dx = dx[:, :, pad:-pad, pad:-pad]  # 3,4번째 차원은 앞뒤로 pad만큼 슬라이싱

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    output_H = (H - pool_height) // stride + 1
    output_W = (W - pool_width) // stride + 1

    out = np.zeros((N, C, output_H, output_W))

    for n in range(N):
        for c in range(C):
            for h in range(output_H):
                for w in range(output_W):
                    h_start = h * stride
                    w_start = w * stride
                    out[n, c, h, w] = np.max(x[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache

    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    output_H = (H - pool_height) // stride + 1
    output_W = (W - pool_width) // stride + 1

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h in range(output_H):
                for w in range(output_W):
                    h_start = h * stride
                    w_start = w * stride

                    # max의 결과였던 값에는 dout이, 아니었던 값에는 0이 더해짐
                    max_index = np.argmax(x[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width])
                    dx[n, c, h_start + (max_index // pool_height), w_start + (max_index % pool_width)] += dout[n, c, h, w]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # x shape -> (N, C)로 만들어야됨
    reshaped_x = np.reshape(x, (-1, x.shape[1]))  # 첫번째 차원은 자동으로 맞추고, 두번째 차원은 C가 되도록 함.

    out, cache = batchnorm_forward(reshaped_x, gamma, beta, bn_param)
    out = np.reshape(out, x.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    reshaped_dout = np.reshape(dout, (-1, dout.shape[1]))  # Shape: (~, C)
    
    dx, dgamma, dbeta = batchnorm_backward(reshaped_dout, cache)
    dx = np.reshape(dx, dout.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer number of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    x = np.reshape(x, (N, G, C // G, H, W))

    group_mean = np.mean(x, axis = (2, 3, 4), keepdims = True)  # Shape: (N, G, 1, 1, 1)
    group_var = np.var(x, axis = (2, 3, 4), keepdims = True)  # Shape: (N, G, 1, 1, 1)

    normalized_x = (x - group_mean) / np.sqrt(group_var + eps)  # Shape: (N, G, 1, 1, 1)
    reshaped_normalized_x = np.reshape(normalized_x, (N, C, H, W))
    out = gamma * reshaped_normalized_x + beta
    cache = (x, group_mean, group_var, reshaped_normalized_x, gamma, beta, G, eps)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, group_mean, group_var, reshaped_normalized_x, gamma, beta, G, eps = cache

    N, C, H, W = dout.shape

    reshaped_x = np.reshape(x, (N, G, C // G, H, W))

    dgamma = np.sum(dout * reshaped_normalized_x, axis = (0, 2, 3), keepdims = True)  # (1, C, 1, 1)
    dbeta = np.sum(dout, axis = (0, 2, 3), keepdims = True)  # (1, C, 1, 1)

    dnormalized_group = dout * gamma  # (N, C, H, W)

    reshaped_dnormalized_group = np.reshape(dnormalized_group, (N, G, C // G, H, W))

    # Shape: (N, G, 1, 1, 1)
    dvar = np.sum(reshaped_dnormalized_group * -0.5 * (reshaped_x - group_mean) * (group_var + eps) ** (-1.5), axis = (2, 3, 4), keepdims = True)

    # Shape: (N, G, 1, 1, 1)
    dmean = np.sum(reshaped_dnormalized_group / -np.sqrt(group_var + eps) + dvar * (reshaped_x - group_mean) * -2 / ((C // G) * H * W), axis = (2, 3, 4), keepdims = True)

    # Shape: (N, C, H, W)
    dx = reshaped_dnormalized_group / np.sqrt(group_var + eps) + dvar * (reshaped_x - group_mean) * 2 / ((C // G) * H * W) + dmean / ((C // G) * H * W)
    dx = np.reshape(dx, (N, C, H, W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
