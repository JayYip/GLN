from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

@add_arg_scope
def conv_layer_norm(inputs,
                  bptr,
                  nclass=None,
                  center=True,
                  scale=True,
                  activation_fn=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None):
    """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.
    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    Can be used as a normalizer function for conv2d and fully_connected.
    Args:
    inputs: a tensor with 2 or more dimensions. The normalization
              occurs over all but the first dimension.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.
    Returns:
    A `Tensor` representing the output of the operation.
    Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
    """   
    with variable_scope.variable_scope(scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        axis = list(range(1, inputs_rank))
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                          'beta')
            beta = variables.model_variable('beta',
                                            shape=params_shape,
                                            dtype=dtype,
                                            initializer=init_ops.zeros_initializer,
                                            collections=beta_collections,
                                            trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections, 'gamma')
            gamma = variables.model_variable('gamma',
                                            shape=params_shape,
                                            dtype=dtype,
                                            initializer=init_ops.ones_initializer(),
                                            collections=gamma_collections,
                                            trainable=trainable)
        # Calculate the moments on the last axis (layer activations).
        bptr_sum = tf.reduce_sum(bptr, [1, 2])
        bptr_sum = tf.expand_dims(tf.expand_dims(bptr_sum, 1), 2)
        norm_weight = tf.div(bptr, bptr_sum)
        mean = tf.mul(norm_weight, inputs)
        mean = tf.reduce_sum(mean, [1, 2])
        mean = tf.expand_dims(tf.expand_dims(mean, 1), 2)
        #print (mean.get_shape().as_list())
        variance = tf.sub(inputs, mean)
        variance = tf.square(variance)
        variance = tf.mul(norm_weight, variance)
        variance = nn.moments(variance, [1, 2], keep_dims=True)[1]
        #print (variance.get_shape().as_list())
        #mean, variance = nn.moments(inputs, axis, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        variance_epsilon = 1E-12
        outputs = nn.batch_normalization(
              inputs, mean, variance, beta, gamma, variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                         sc.original_name_scope,
                                         outputs)
