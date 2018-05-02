from __future__ import division

import numpy as np
import tensorflow as tf

# components
from tensorflow.python.ops.nn import dropout as drop
from proposal.cnn import conv_layer as conv
from proposal.cnn import conv_relu_layer as conv_relu
from proposal.cnn import pooling_layer as pool
from proposal.cnn import fc_layer as fc
from proposal.cnn import fc_relu_layer as fc_relu

def vs_multilayer(input_batch,name,middle_layer_dim=1000,reuse=False,test=False):
    with tf.variable_scope(name):
        if reuse==True:
            # print name+" reuse variables"
            tf.get_variable_scope().reuse_variables()
        else:
            pass
            # print name+" doesn't reuse variables"

        layer1 = fc_relu('layer1', input_batch, output_dim=middle_layer_dim)
        if test:
            layer1 = drop(layer1, 1)
        else:
            layer1=drop(layer1,0.5)
        outputs = fc('layer2', layer1,output_dim=4)
    return outputs
