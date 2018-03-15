import tensorflow as tf 
import math
import pdb
from six.moves import xrange
SEED = 66478
def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)
def init():
    beta = tf.get_variable('beta', [1,64], tf.float32,initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', [1,64], tf.float32,initializer=math.sqrt(2 / (9.0 * 64)) * tf.random_normal([1,64]))
    moving_mean = tf.get_variable('moving_mean', [1,64], tf.float32,initializer=tf.constant_initializer(0.0, tf.float32))
    moving_variance = tf.get_variable('moving_variance', [1,64], tf.float32,initializer=tf.constant_initializer(1.0, tf.float32))
    return None
#def batch_normalization(logits,sess, name="bn",phase_train=True):
#def batch_normalization(logits, scale, offset, pop_mean,pop_var,isCovNet = True, name="bn",phase_train=False,decay = 0.999):
##    pop_mean = tf.get_variable(tf.zeros([logits.get_shape()[-1]]), trainable=False,name='bn_mean')
##    pop_var = tf.get_variable(0.01*tf.ones([logits.get_shape()[-1]]), trainable=False,name='bn_var')
#    if phase_train:
#        if isCovNet:
#            batch_mean, batch_var = tf.nn.moments(logits,[0,1,2])
#        else:
#            batch_mean, batch_var = tf.nn.moments(logits,[0])   
#
#        train_mean = tf.assign(pop_mean,
#                               decay * pop_mean + batch_mean * (1 - decay))
#        train_var = tf.assign(pop_var,
#                              decay * pop_var + batch_var * (1 - decay))
#        with tf.control_dependencies([train_mean, train_var]):
##            return tf.nn.batch_normalization(logits,
##                pop_mean, pop_var, offset, scale, 0)
#            return tf.nn.batch_normalization(logits,
#                batch_mean, batch_var, offset, scale,0)
#    else:
#        return tf.nn.batch_normalization(logits,
#            pop_mean, pop_var, offset, scale, 0)
#        return tf.nn.batch_normalization(logits,
#            pop_mean, pop_var, offset, scale, 0.001)
#        return tf.contrib.layers.batch_norm(inputs=logits,decay = 0.999,center = True,scale = True, is_training=False,reuse=True,scope ="bn")

def get_conv_weights(weight_shape, sess, name="get_conv_weights"):
#	number = math.sqrt(1.0 / (weight_shape[0] * weight_shape[1] * (weight_shape[2] +  weight_shape[3]))) * sess.run(tf.random_normal(weight_shape))
#	pdb.set_trace()
#	return math.sqrt(2.0 / max(weight_shape[2],weight_shape[3])) * sess.run(tf.random_normal(weight_shape))
	return math.sqrt(2.0 / (9 * 64)) * sess.run(tf.random_normal(weight_shape))
def get_bn_weights(weight_shape, sess, name="get_bn_weights"):
	weights = get_conv_weights(weight_shape, sess)

	return weights

#def get_bn_weights(weight_shape, clip_b, sess, name="get_bn_weights"):
#	weights = get_conv_weights(weight_shape, sess)
#	return weights
#	return clipping(weights, clip_b)
def clipping(A, clip_b, name="clipping"):
	h, w = A.shape 
	for i in xrange(h):
		for j in xrange(w):
			if A[i,j] >= 0 and A[i,j] < clip_b:
				A[i,j] = clip_b
			elif A[i,j] > -clip_b and A[i,j] < 0:
				A[i,j] = -clip_b
	return A