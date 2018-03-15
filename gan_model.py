'''
DnCNN
paper: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising

a tensorflow version of the network DnCNN
just for personal exercise

author: momogary
'''
import pdb
import tensorflow as tf 
import numpy as np 
import math, os
from glob import glob
from ops import *
from utils import *
from six.moves import xrange
import time
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow  
from PIL import Image
class DnCNN(object):
	def __init__(self, sess, image_size=40, batch_size=128,
					output_size=40, input_c_dim=1, output_c_dim=1, 
					sigma=25, clip_b=0.025, lr=0.01, epoch=60,
					ckpt_dir='./checkpoint', sample_dir='./sample',
					test_save_dir='./test',
					dataset='BSD400', testset='Set12'):
		self.sess = sess
		self.is_gray = (input_c_dim == 1)
		self.batch_size = batch_size
		self.image_size = image_size
		self.output_size = output_size
		self.input_c_dim = input_c_dim
		self.output_c_dim = output_c_dim
		self.sigma = sigma
		self.clip_b = clip_b
		self.lr = lr
		self.numEpoch = epoch
		self.ckpt_dir = ckpt_dir
		self.trainset = dataset
		self.testset = testset
		self.sample_dir = sample_dir
		self.test_save_dir = test_save_dir
		self.epoch = epoch
		self.save_every_iter = 500
		self.eval_every_iter = 5400
		# Adam setting (default setting)
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.alpha = 0.01
		self.epsilon = 1e-8
#		self.W_params = list([])
#		self.b_params = list([])
		self.bn_params = list([])
		self.reuse = False
		self.build_model(self.reuse)

#		self.var_params = list([])
	def generator(self,g_input,reuse):
#		self.X = tf.placeholder(tf.float32, \
#					[None, self.image_size, self.image_size, self.input_c_dim], \
#					name='noisy_image')
		with tf.variable_scope("generator", reuse=reuse):
			with tf.variable_scope('conv1', reuse=reuse):
				layer_1_output,layer_1_output_concat = self.per_layer(g_input, self.input_c_dim , useBN = False, use_b = True)
#		 layer 2 to 16
			with tf.variable_scope('conv2', reuse=reuse):
				layer_2_output,layer_2_output_concat = self.per_layer(layer_1_output, input_c_dim = 64, useBN = True, use_b = False )
			with tf.variable_scope('conv3', reuse=reuse):
				layer_3_output,layer_3_output_concat = self.per_layer(layer_2_output, input_c_dim = 64, useBN = True, use_b = False )
			with tf.variable_scope('conv4', reuse=reuse):
				layer_4_output,layer_4_output_concat = self.per_layer(layer_3_output, input_c_dim = 64, useBN = True, use_b = False )
			with tf.variable_scope('conv5', reuse=reuse):
				layer_5_output,layer_5_output_concat = self.per_layer(layer_4_output, input_c_dim = 64, useBN = True, use_b = False )
			with tf.variable_scope('conv6', reuse=reuse):
				layer_6_output,layer_6_output_concat = self.per_layer(layer_5_output, input_c_dim = 64, useBN = True, use_b = False )          
			with tf.variable_scope('conv7_1', reuse=reuse):
				layer_7_1_output,layer_7_1_output_concat = self.per_layer(layer_6_output, input_c_dim = 64, useBN = True, use_b = False )
			with tf.variable_scope('conv8_1', reuse=reuse):
				layer_8_1_output,layer_8_1_output_concat = self.per_layer(layer_7_1_output, input_c_dim = 64, useBN = True, use_b = False )   
			with tf.variable_scope('conv9_1', reuse=reuse):
				self.Y_1 = self.conv_layer(layer_8_1_output, [3, 3, 64, self.output_c_dim],  b_init = 0.0, stridemode=[1,1,1,1],use_b = True)
			with tf.variable_scope('conv7_2', reuse=reuse):
				layer_7_2_output,layer_7_2_output_concat = self.per_layer(layer_6_output, input_c_dim = 64, useBN = True, use_b = False )
			with tf.variable_scope('conv8_2', reuse=reuse):
				layer_8_2_output,layer_8_2_output_concat = self.per_layer(layer_7_2_output, input_c_dim = 64, useBN = True, use_b = False )   
			with tf.variable_scope('conv9_2', reuse=reuse):
				self.Y_2 = self.conv_layer(layer_8_2_output, [3, 3, 64, self.output_c_dim],  b_init = 0.0, stridemode=[1,1,1,1],use_b = True)
			with tf.variable_scope('conv7_3', reuse=reuse):
				layer_7_3_output,layer_7_3_output_concat = self.per_layer(layer_6_output, input_c_dim = 64, useBN = True, use_b = False )
			with tf.variable_scope('conv8_3', reuse=reuse):
				layer_8_3_output,layer_8_3_output_concat = self.per_layer(layer_7_3_output, input_c_dim = 64, useBN = True, use_b = False )   
			with tf.variable_scope('conv9_3', reuse=reuse):
				self.Y_3 = self.conv_layer(layer_8_3_output, [3, 3, 64, self.output_c_dim],  b_init = 0.0, stridemode=[1,1,1,1],use_b = True)

			with tf.variable_scope('theta', reuse=reuse):
				theta1 = tf.get_variable("theta1",[1])
				theta2 = tf.get_variable("theta2",[1])
				theta3 = tf.get_variable("theta3",[1])

#			theta4 = tf.get_variable("theta4",[1])
				self.Y = theta1 * self.Y_1 + theta2 * self.Y_2 + theta3 * self.Y_3        
		return self.Y
	def discriminator(self,d_input,reuse,useBN=True,use_b=False):	
		with tf.variable_scope("discriminator", reuse=reuse):
			with tf.variable_scope('d_conv1', reuse=reuse):
				d_layer_1_output = self.layer(d_input, filter_shape = [3, 3, self.output_c_dim, 64], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)
			with tf.variable_scope('d_conv2', reuse=reuse):
				d_layer_2_output = self.layer(d_layer_1_output, filter_shape = [3, 3, 64, 32], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)
			with tf.variable_scope('d_conv3', reuse=reuse):
				d_layer_3_output = self.layer(d_layer_2_output, filter_shape = [3, 3, 32, 16], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)
			with tf.variable_scope('d_conv4', reuse=reuse):
				d_layer_4_output = self.layer(d_layer_3_output, filter_shape = [3, 3, 16, 8], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)
			d_layer_4_output = tf.contrib.layers.flatten(d_layer_4_output)
			with tf.variable_scope('d_conv5', reuse=reuse):
				d_layer_5_output = tf.layers.dense(d_layer_4_output, units = 1)   
#			d_output = tf.nn.softmax(d_layer_5_output)
		return d_layer_5_output
	def build_model(self,reuse):
		# input : [batchsize, image_size, image_size, channel]
		self.X = tf.placeholder(tf.float32, \
					[None, self.image_size, self.image_size, self.input_c_dim], \
					name='noisy_image')
		self.X_ = tf.placeholder(tf.float32, \
					[None, self.image_size, self.image_size, self.input_c_dim], \
					name='clean_image')
#		self.X_clean = tf.placeholder(tf.float32, \
#					[None, self.image_size, self.image_size, self.input_c_dim], \
#					name='input_clean_image')
		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		self.Y_ = self.X - self.X_ # noisy image - clean image
#		self.loss_l2 = (1.0 / self.batch_size) * tf.nn.l2_loss(self.Y - self.Y_)
#			theta4 = tf.get_variable("theta4",[1])
		self.Y = self.generator(g_input=self.X,reuse=False)
		self.g_d_output = self.discriminator(d_input=self.Y, reuse=False)
		self.noise_d_output = self.discriminator(d_input=self.Y_, reuse=True)
		self.mse_loss = 0.5*(1.0 / self.batch_size) * tf.reduce_sum(tf.square(self.Y - self.Y_))
		self.g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_output,labels = tf.ones_like(self.g_d_output)))
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.noise_d_output,labels = tf.ones_like(self.noise_d_output)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_output,labels = tf.zeros_like(self.g_d_output)))
		self.d_loss_all = self.d_loss_real + self.d_loss_fake 
		self.g_loss_all = self.g_loss_fake *0.0066 + self.mse_loss  
		self.l2 = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		t_vars = tf.trainable_variables()
		g_vars = [var for var in t_vars if 'generator' in var.name]
		d_vars = [var for var in t_vars if 'discriminator' in var.name]
  #		self.loss = self.loss 
		print(d_vars)
#		pdb.set_trace()
		self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='AdamOptimizer').minimize(self.g_loss_all,var_list=g_vars)
		self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='AdamOptimizer2').minimize(self.d_loss_all,var_list=d_vars)
#		optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='GD')
#		self.train_step_2 = optimizer_2.minimize(self.loss , var_list=self.bn_params)
		# create this init op after all variables specified
		self.init = tf.global_variables_initializer() 
		self.saver = tf.train.Saver(max_to_keep=100000)
		print("[*] Initialize model successfully...")
		self.writer = tf.summary.FileWriter("./log",tf.get_default_graph())
		self.writer.close()


	def conv_layer(self, inputdata, weightshape, b_init, stridemode,use_b = False):
		# weights
			W = tf.get_variable('weights', regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0001) ,shape = weightshape, initializer = \
							tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
			conv = tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME")   
			if use_b:
				b = tf.get_variable('biases', [weightshape[-1]], initializer = \
							tf.constant_initializer(b_init))
#			tf.summary.histogram('histogram_b', b)
#			tf.summary.histogram('histogram_W', W)

				output = tf.nn.bias_add(conv, b)
				return output
			return conv # SAME with zero padding

	def deconv_layer(self, inputdata, weightshape, b_init, stridemode,out_shape,use_b = False):
		# weights
			W = tf.get_variable('weights', regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0001) ,shape = weightshape, initializer = \
							tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
			conv = tf.nn.conv2d_transpose(inputdata, W, output_shape = out_shape,strides=stridemode, padding="SAME")   
			if use_b:
				b = tf.get_variable('biases', [weightshape[-1]], initializer = \
							tf.constant_initializer(b_init))
#			tf.summary.histogram('histogram_b', b)
#			tf.summary.histogram('histogram_W', W)

				output = tf.nn.bias_add(conv, b)
				return output
			return conv # SAME with zero padding
   
	def bn_layer(self, logits, output_dim, b_init = 0.0):
		alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer = \
								tf.constant_initializer(get_bn_weights([1, output_dim], self.sess)))
		beta = tf.get_variable('bn_beta', [1, output_dim], initializer = \
								tf.constant_initializer(0.0))
		pop_mean = tf.get_variable('bn_mean', [1,output_dim], tf.float32,initializer=tf.constant_initializer(0.0, tf.float32),trainable = False)
		pop_var = tf.get_variable('bn_variance', [1,output_dim], tf.float32,initializer=tf.constant_initializer(0.01, tf.float32),trainable = False)

		return self.batch_normalization(logits, alpha, beta,pop_mean,pop_var, isCovNet = True )
#  
	def batch_normalization(self,logits, scale, offset, pop_mean,pop_var,isCovNet = True, name="bn",phase_train=True,decay = 0.999):
#    pop_mean = tf.get_variable(tf.zeros([logits.get_shape()[-1]]), trainable=False,name='bn_mean')
#    pop_var = tf.get_variable(0.01*tf.ones([logits.get_shape()[-1]]), trainable=False,name='bn_var')
		if phase_train:
			if isCovNet:
					batch_mean, batch_var = tf.nn.moments(logits,[0,1,2])
			else:
					batch_mean, batch_var = tf.nn.moments(logits,[0])   
			
			train_mean = tf.assign(pop_mean,
                               decay * pop_mean + batch_mean * (1 - decay))
			train_var = tf.assign(pop_var,
                              decay * pop_var + batch_var * (1 - decay))
#			if 'conv15' in pop_mean.name:
#					pdb.set_trace()
#					self.bn_mean = pop_mean
#					self.bn_var = pop_var
#					self.bn_mean_new = batch_mean
#					self.bn_var_new = batch_var
			with tf.control_dependencies([train_mean, train_var]):
#            return tf.nn.batch_normalization(logits,
#                pop_mean, pop_var, offset, scale, 0)
                            return tf.nn.batch_normalization(logits,
                batch_mean, batch_var, offset, scale,0)
                            
		else:
#			if 'conv15' in pop_mean.name:
#					pdb.set_trace()
#					self.bn_mean = pop_mean
#					self.bn_var = pop_var
			return tf.nn.batch_normalization(logits,
            pop_mean, pop_var, offset, scale, 0)
	def per_layer(self, inputdata, input_c_dim,useBN=False, use_b = False):
#		with tf.variable_scope('1x1'):
#			output_1x1 = self.layer(inputdata, [1, 1, input_c_dim, 8], useBN=useBN, use_b = use_b)     
		with tf.variable_scope('3x3'):
			output_3x3 = self.layer(inputdata, filter_shape = [3, 3, input_c_dim, 64], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)   
#			with tf.variable_scope('3x3_1'):
#				output_3x3_1 = self.layer(inputdata, [1, 3, input_c_dim, 64], useBN=useBN, use_b = use_b)  
#			with tf.variable_scope('3x3_2'):
#				output_3x3 = self.layer(output_3x3_1, [3, 1, 64, 64], useBN=useBN, use_b = use_b)     
		with tf.variable_scope('5x5'):
			with tf.variable_scope('5x5_1'):
				output_5x5_1 = self.layer(inputdata, filter_shape =[1, 5, input_c_dim, 48], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)  
			with tf.variable_scope('5x5_2'):
				output_5x5 = self.layer(output_5x5_1,filter_shape = [5, 1, 48, 48], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)     
		with tf.variable_scope('7x7'):
			with tf.variable_scope('7x7_1'):
				output_7x7_1 = self.layer(inputdata, filter_shape = [1, 7, input_c_dim, 16], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)
			with tf.variable_scope('7x7_2'):
				output_7x7 = self.layer(output_7x7_1,filter_shape = [7, 1, 16, 16], b_init = 0.0, stridemode=[1,1,1,1], useBN=useBN, use_b = use_b)     
#		concat_output=tf.concat([output_1x1,output_3x3,output_5x5,output_7x7],axis=3)
		concat_output=tf.concat([output_3x3,output_5x5,output_7x7],axis=3)
		with tf.variable_scope('reduce'):
#			with tf.variable_scope('3x3_1'):
#				output_3x3_2 = self.layer(concat_output, [1, 3, 128, 64], useBN=useBN, use_b = use_b)  
#			with tf.variable_scope('3x3_2'):
#				output = self.layer(output_3x3_2, [3, 1, 64, 64], useBN=useBN, use_b = use_b)         
			output = self.layer(concat_output, [3, 3, 128, 64], useBN=useBN, use_b = use_b)
		return output,concat_output  

  
	def layer(self, inputdata, filter_shape, b_init = 0.0, stridemode=[1,1,1,1], useBN = True,use_b = False , conv = True , out_shape = None):
		if conv:
#			output = tf.nn.relu(batch_normalization(logits,self.sess))
			logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode,use_b)
#			output = tf.nn.relu(logits)
		else:
			logits = self.deconv_layer(inputdata, filter_shape, b_init, stridemode,out_shape,use_b)
		if useBN:
#			output = tf.nn.relu(batch_normalization(logits,self.sess))
			output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1],self.sess))
#			output = tf.nn.relu(logits)
		else:
			output = tf.nn.relu(logits)
		return output
	def mn(self,l):
		if len(l)>20:
			l_last = l[len(l)-20:len(l)]
		else:
			l_last = l
		if sum(l_last) == 0:
			return 0
		return sum(l_last) / len(l_last)
	def train(self):
		# init the variables
		f = file('/home/lyj/tensorflow_code/DnCNN-tensorflow-master/perlayer_reduce3*3_8layer_data_agumentation_8.txt','w+')
		self.sess.run(self.init)
#		finetune = self.load('./checkpoint')
		# get data
		test_files = glob('./test/*.png'.format(self.testset))
		test_data = load_images(test_files) # list of array of different size, 4-D
		data = load_data(filepath='./img_clean_pats.npy')
#		noise_data = add_noise(data, self.sigma, self.sess)
		numBatch = int(data.shape[0] / self.batch_size)
		print(data.shape[0])
		print(self.epoch)
		print(numBatch)
		print("[*] Data shape = " + str(data.shape))
#		if self.reuse==False:  
#			if self.load(self.ckpt_dir):
#				print(" [*] Load SUCCESS")
#			else:
#				print(" [!] Load failed...")
		counter = 0
		print("[*] Start training : ")
		this_time = time.time()
		mse_costs = []
		g_loss_costs = []
		d_loss_costs = []
#		net1_costs = []
#		net2_costs = []
#		net3_costs = []
#		net4_costs = []
		for epoch in xrange(self.epoch):
			for batch_id in xrange(numBatch):
				batch_images = data[batch_id*self.batch_size:(batch_id+1)*self.batch_size, :, :, :]
#				print(batch_images.shape)
#				pdb.set_trace()
				train_images = batch_images + self.sigma / 255.0 * np.random.normal(loc=0, scale=1,size=batch_images.shape)
#				pdb.set_trace()
#				train_images = noise_data[batch_id*self.batch_size:(batch_id+1)*self.batch_size, :, :, :]
#				theta_1 = math.pow(0.95,epoch)
#				theta_2 = math.pow(0.975,epoch)    
				if epoch < 25:
					_, mse_loss ,g_loss,output = self.sess.run([self.g_optimizer,self.mse_loss,self.g_loss_fake,self.g_d_output],\
                              feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.001})
					if batch_id % 2 == 0:
						_, mse_loss ,d_loss= self.sess.run([self.d_optimizer,self.mse_loss,self.d_loss_all], \
						  feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.001})
				elif epoch >= 25 and epoch < 50:
					_,   mse_loss ,g_loss= self.sess.run([self.g_optimizer,self.mse_loss,self.g_loss_fake], \
						feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.0001})
					if batch_id % 2 == 0:
						_, mse_loss ,d_loss= self.sess.run([self.d_optimizer,self.mse_loss,self.d_loss_all], \
						  feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.0001})
				elif epoch >= 50 and epoch < 55:
					_, mse_loss ,g_loss= self.sess.run([self.g_optimizer,self.mse_loss,self.g_loss_fake], \
						feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.00001})
					if batch_id % 2 == 0:
						_, mse_loss ,d_loss= self.sess.run([self.d_optimizer,self.mse_loss,self.d_loss_all], \
						  feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.00001})
				elif epoch >= 55:
					_, mse_loss ,g_loss= self.sess.run([self.g_optimizer,self.mse_loss,self.g_loss_fake], \
						feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.000001})
					if batch_id % 2 == 0:
						_, mse_loss ,d_loss= self.sess.run([self.d_optimizer,self.mse_loss,self.d_loss_all], \
						  feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.000001})
				print(output)
				mse_costs.append(mse_loss)
				g_loss_costs.append(g_loss)
				d_loss_costs.append(d_loss)
				last_time = this_time
				this_time = time.time()
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, mse_loss: %.6f, g_loss: %.6f, d_loss: %.6f" \
					% (epoch + 1, batch_id + 1, numBatch,
						time.time() - last_time, self.mn(mse_costs), self.mn(g_loss_costs), self.mn(d_loss_costs)))

				#print(test_files)
				#print(test_data)
				counter += 1
				f.write("Epoch: [%2d] [%4d/%4d] time: %4.4f, mse_loss: %.6f, g_loss: %.6f, d_loss: %.6f" \
					% (epoch + 1, batch_id + 1, numBatch,
						time.time() - last_time, self.mn(mse_costs), self.mn(g_loss_costs), self.mn(d_loss_costs)))
				f.write("\n")
#				self.writer.add_summary(summary, epoch*1600+batch_id)
#				if np.mod(counter, self.eval_every_iter) == 0:
#					self.evaluate(epoch, counter, test_data,test_files)
				# save the model
				if np.mod(counter, self.save_every_iter) == 0:
					self.save(counter)
#			self.writer.add_summary(summary, epoch)
		print("[*] Finish training.")
		f.close()
		self.writer.close()
  
	def save(self,counter):
		model_name = "DnCNN.model"
		model_dir = "gan_denoise_theta_%s_%s_%s" % (self.trainset, \
									self.batch_size, self.image_size)
		checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		print("[*] Saving model...")
		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=counter)

	def sampler(self,X_test):

		with tf.variable_scope('conv1', reuse=True):
			layer_1_output,layer_1_output_concat = self.per_layer(X_test, self.input_c_dim , useBN = False, use_b = True)
#		 layer 2 to 16
		with tf.variable_scope('conv2', reuse=True):
			layer_2_output,layer_2_output_concat = self.per_layer(layer_1_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv3', reuse=True):
			layer_3_output,layer_3_output_concat = self.per_layer(layer_2_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv4', reuse=True):
			layer_4_output,layer_4_output_concat = self.per_layer(layer_3_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv5', reuse=True):
			layer_5_output,layer_5_output_concat = self.per_layer(layer_4_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv6', reuse=True):
			layer_6_output,layer_6_output_concat = self.per_layer(layer_5_output, input_c_dim = 64, useBN = True, use_b = False )      
		with tf.variable_scope('conv7', reuse=True):
			layer_7_output,layer_7_output_concat = self.per_layer(layer_6_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv8', reuse=True):
			layer_8_output,layer_8_output_concat = self.per_layer(layer_7_output, input_c_dim = 64, useBN = True, use_b = False )   
		with tf.variable_scope('conv9', reuse=True):
			layer_9_output,layer_9_output_concat = self.per_layer(layer_8_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv10', reuse=True):
			layer_10_output,layer_10_output_concat = self.per_layer(layer_9_output, input_c_dim = 64, useBN = True, use_b = False )      
		with tf.variable_scope('conv11_1', reuse=True):
			layer_11_1_output,layer_11_1_output_concat = self.per_layer(layer_10_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv12_1', reuse=True):
			layer_12_1_output,layer_12_1_output_concat = self.per_layer(layer_11_1_output, input_c_dim = 64, useBN = True, use_b = False )   
		with tf.variable_scope('conv13_1', reuse=True):
			Y_1 = self.conv_layer(layer_12_1_output, [3, 3, 64, self.output_c_dim],  b_init = 0.0, stridemode=[1,1,1,1],use_b = True)
		with tf.variable_scope('conv11_2', reuse=True):
			layer_11_2_output,layer_11_2_output_concat = self.per_layer(layer_10_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv12_2', reuse=True):
			layer_12_2_output,layer_12_2_output_concat = self.per_layer(layer_11_2_output, input_c_dim = 64, useBN = True, use_b = False )   
		with tf.variable_scope('conv13_2', reuse=True):
			Y_2 = self.conv_layer(layer_12_2_output, [3, 3, 64, self.output_c_dim],  b_init = 0.0, stridemode=[1,1,1,1],use_b = True)
		with tf.variable_scope('conv11_3', reuse=True):
			layer_11_3_output,layer_11_3_output_concat = self.per_layer(layer_10_output, input_c_dim = 64, useBN = True, use_b = False )
		with tf.variable_scope('conv12_3', reuse=True):
			layer_12_3_output,layer_12_3_output_concat = self.per_layer(layer_11_3_output, input_c_dim = 64, useBN = True, use_b = False )   
		with tf.variable_scope('conv13_3', reuse=True):
			Y_3 = self.conv_layer(layer_12_3_output, [3, 3, 64, self.output_c_dim],  b_init = 0.0, stridemode=[1,1,1,1],use_b = True)
		with tf.variable_scope('theta', reuse=True):
			theta1 = tf.get_variable("theta1",[1])
			theta2 = tf.get_variable("theta2",[1])
			theta3 = tf.get_variable("theta3",[1])
#			theta4 = tf.get_variable("theta4",[1])
		self.Y = theta1 * Y_1 + theta2 * Y_2 + theta3 * Y_3 
   #tf.get_variable_scope().reuse_variables()
#		with tf.variable_scope('conv1', reuse=True):
#			layer_1_output,layer_1_output_concat = self.per_layer(X_test, self.input_c_dim , useBN = False, use_b = True)
#		# layer 2 to 16
#		with tf.variable_scope('conv2', reuse=True):
#			layer_2_output,layer_2_output_concat = self.per_layer(layer_1_output, input_c_dim = 64, useBN = True, use_b = False )
#		with tf.variable_scope('conv3', reuse=True):
#			layer_3_output,layer_3_output_concat = self.per_layer(layer_2_output, input_c_dim = 64, useBN = True, use_b = False )
#		with tf.variable_scope('conv4', reuse=True):
#			layer_4_output,layer_4_output_concat = self.per_layer(layer_3_output, input_c_dim = 64, useBN = True, use_b = False )
#		with tf.variable_scope('conv5', reuse=True):
#			layer_5_output,layer_5_output_concat = self.per_layer(layer_4_output, input_c_dim = 64, useBN = True, use_b = False )
#		with tf.variable_scope('conv6', reuse=True):
#			layer_6_output,layer_6_output_concat = self.per_layer(layer_5_output, input_c_dim = 64, useBN = True, use_b = False )
#		with tf.variable_scope('conv7', reuse=True):
#			layer_7_output,layer_7_output_concat = self.per_layer(layer_6_output, input_c_dim = 64, useBN = True, use_b = False )
#		with tf.variable_scope('conv8', reuse=True):
#			layer_8_output,layer_8_output_concat = self.per_layer(layer_7_output, input_c_dim = 64, useBN = True, use_b = False )
##		with tf.variable_scope('conv9', reuse=reuse):
##			layer_9_output,layer_9_output_concat = self.per_layer(layer_8_output, input_c_dim = 64, useBN = True, use_b = False )
##		with tf.variable_scope('conv10', reuse=reuse):
##			layer_10_output,layer_10_output_concat = self.per_layer(layer_9_output, input_c_dim = 64, useBN = True, use_b = False )
##		with tf.variable_scope('conv11', reuse=reuse):
##			layer_11_output,layer_11_output_concat = self.per_layer(layer_10_output, input_c_dim = 64, useBN = True, use_b = False )
##		with tf.variable_scope('conv12', reuse=reuse):
##			layer_12_output,layer_12_output_concat = self.per_layer(layer_11_output, input_c_dim = 64, useBN = True, use_b = False )
##		with tf.variable_scope('conv13', reuse=reuse):
##			layer_13_output,layer_13_output_concat = self.per_layer(layer_12_output, input_c_dim = 64, useBN = True, use_b = False )
##		with tf.variable_scope('conv14', reuse=reuse):
##			layer_14_output,layer_14_output_concat = self.per_layer(layer_13_output, input_c_dim = 64, useBN = True, use_b = False )
##		with tf.variable_scope('conv15', reuse=reuse):
##			layer_15_output,layer_15_output_concat = self.per_layer(layer_14_output, input_c_dim = 64, useBN = True, use_b = False )
##		with tf.variable_scope('conv16', reuse=reuse):
##			layer_16_output,layer_16_output_concat = self.per_layer(layer_15_output, input_c_dim = 64, useBN = True, use_b = False )
		output=tf.reduce_mean(self.Y,axis=3)
		return output,layer_1_output
                       
                       
	def load(self, checkpoint_dir):
		'''Load checkpoint file'''
		print("[*] Reading checkpoint...")

		model_dir = "perlayer_reduce3*3_8layer_data_agumentation_%s_%s_%s" % (self.trainset, self.batch_size, self.image_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		print(checkpoint_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False
	def show(self,data,i):
#		data=self.layer_1_output
		data = (data - data.min()) / (data.max() - data.min())
		data = data.transpose(3,1,2,0)
#		pdb.set_trace()
        # force the number of filters to be square
		n = int(np.ceil(np.sqrt(data.shape[0])))
		padding = (((0, n ** 2 - data.shape[0]), (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
		data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
        
        # tile the filters into an image
		data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
		data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

		cat_image = np.clip(data[:,:,0] * 255, 0, 255).astype('uint8')
		im = Image.fromarray(cat_image.astype('uint8')).convert('L')
		im.save(os.path.join(self.sample_dir, 'init_fearture16_8th_concat_iter_%d.png' %(i) ), 'png')  
#		for i in range(1,128):d
#        		plt.figure()
#        		plt.imshow(data[:,:,i], cmap ='gray'); plt.axis('off')
#		pdb.set_trace()  
		return None

	def test(self):
		"""Test DnCNN"""
		f = file('/home/lyj/tensorflow_code/DnCNN-tensorflow-master/test_per_8layer_4_sigma_25_multi_ensemable.txt','w+')
		#merged = tf.summary.merge_all()
		# init variables
		tf.initialize_all_variables().run()
		test_files = glob('./test/*.png'.format(self.testset))
#		test_files = glob(os.path.join(self.test_save_dir, '{}/*.png'.format(self.testset)))
		print(test_files)
		counter = 0  
		# load testing input
		X_test = tf.placeholder(tf.float32, \
					[1,None,None,1], name='noisy_image_test')
		predicted_noise,layer_1_output = self.sampler(X_test)
		print("[*] Loading test images ...")
		test_data = load_images(test_files) # list of array of different size
		for order in range(160,192): 		
        		model_path = './checkpoint/per_8layer_end_3layer_25_half_ensemable_BSD400_128_40/DnCNN.model-%d'%(order*500)
        		saver = tf.train.Saver()
        		saver.restore(self.sess, model_path)
        		psnr_sum = 0
        		psnr_init_sum = 0 
          	     
        		for idx in xrange(len(test_files)):
        			noisy_image = add_noise(1 / 255.0 * test_data[idx], self.sigma, self.sess)  # ndarray
#        			pdb.set_trace()
        			noise_ = (noisy_image  - 1 / 255.0 * test_data[idx]) * 255.0  # ndarray
#        			noise_var = np.var(noise_[:,:,:,0])
#        			noise_mean = np.mean(noise_[:,:,:,0])
#        			print("noise_var: %4f" %(noise_var))
#        			print("noise_mean: %4f" %(noise_mean))
#        			plt.figure()
#        			image = noisy_image[0,:,:,0]
#        			plt.imshow(image, cmap ='gray')
#        			pdb.set_trace()        
        			predicted_noise_,layer_1_output_ = self.sess.run([predicted_noise,layer_1_output],feed_dict={X_test : noisy_image})
#        			bn_mean,bn_var,bn_mean_new,bn_var_new = self.sess.run([self.bn_mean,self.bn_var,self.bn_mean_new,self.bn_var_new],feed_dict={X_test : noisy_image})
#        			bn_mean,bn_var = self.sess.run([self.bn_mean,self.bn_var],feed_dict={X_test : noisy_image})
#        			show_ = self.show(layer_1_output_,order)

        			counter = counter + 1
#        			predicted_noise_=predicted_noise_*255
#        			predicted_noise_var = np.var(predicted_noise_)
#        			predicted_noise_mean = np.mean(predicted_noise_)
#        			print("predicted_noise_var: %4f" %(predicted_noise_var))
#        			print("predicted_noise_mean: %4f" %(predicted_noise_mean))
        			noisy_image=noisy_image[:,:,:,0]
#        			pdb.set_trace()
        			output_clean_image = (noisy_image - predicted_noise_) * 255
#        			output_clean_image = predicted_noise_ * 255
 			# calculate PSNR
        			groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
        			groundtruth=groundtruth[:,:,:,0]
        			noisyimage = np.clip(noisy_image * 255, 0, 255).astype('uint8')
        			outputimage = np.clip(output_clean_image, 0, 255).astype('uint8')
        			psnr = cal_psnr(groundtruth, outputimage)
        			print(psnr)
        			psnr_init_ = cal_psnr(groundtruth, noisyimage)
        			print(psnr_init_)
        			psnr_init_sum += psnr_init_   
        			psnr_sum += psnr
        			save_images(groundtruth, noisyimage, outputimage, os.path.join(self.sample_dir, 'test_mean_%d_%d.png' % (idx,counter)))
        		avg_psnr = psnr_sum / len(test_files)
        		avg_psnr_init = psnr_init_sum / len(test_files)
        		#tf.summary.scalar('Average PSNR', avg_psnr)
        		print("--- Test ---- Average PSNR %.4f ---" % avg_psnr)
        		print("--- Test ---- Average init PSNR %.4f ---" % avg_psnr_init)
        		f.write("--- Test ---- Average PSNR %.4f ---" % avg_psnr)
        		f.write("\n")
		f.close()          
	def evaluate(self, epoch, counter, test_data,test_files):
		'''Evaluate the model during training'''
		#print("[*] Evaluating...")
		#print(test_data)
		psnr_sum = 0
		psnr_init_sum = 0  
		for idx in xrange(len(test_files)):
			noisy_image = add_noise(1 / 255.0 * test_data[idx], self.sigma, self.sess)  # ndarray
			predicted_noise = self.forward(noisy_image)
			output_clean_image = (noisy_image - predicted_noise) * 255

			groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
			noisyimage = np.clip(noisy_image * 255, 0, 255).astype('uint8')
			outputimage = np.clip(output_clean_image, 0, 255).astype('uint8')
			print(groundtruth.shape)
			print(noisyimage.shape)
			print(outputimage.shape)
#			print("groundtruth")   
#			plt.figure()
#			groundtruth_ = groundtruth[0,:,:,0]
#			print(groundtruth_)
#			plt.imshow(groundtruth_, cmap ='gray')
#			print("noisyimage")    
#   			plt.figure()
#			noisyimage_ = noisyimage[0,:,:,0]
#			print(noisyimage_)
#			plt.imshow(noisyimage_, cmap ='gray')
#			print("outputimage") 
#			plt.figure()
#			outputimage_ = outputimage[0,:,:,0]
#			print(outputimage_)
#			plt.imshow(outputimage_, cmap ='gray')
 			# calculate PSNR
			psnr = cal_psnr(groundtruth, outputimage)
			print(psnr)
			psnr_init_ = cal_psnr(groundtruth, noisyimage)
			print(psnr_init_)
			psnr_init_sum += psnr_init_   
			psnr_sum += psnr
			save_images(groundtruth, noisyimage, outputimage, \
						os.path.join(self.sample_dir, 'test%d_%d_%d.png' % (idx, epoch, counter)))
		avg_psnr = psnr_sum / len(test_files)
		avg_psnr_init = psnr_init_sum / len(test_files)  
		print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
		print("--- Test ---- Average init PSNR %.2f ---" % avg_psnr_init)
	def read_ckt(self):

		model_dir = "bn_init_l2_%s_%s_%s" % (self.trainset, \
									self.batch_size, self.image_size)
		checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)
		reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_dir)  
		var_to_shape_map = reader.get_variable_to_shape_map()  
		for key in var_to_shape_map:  
			print("tensor_name: ", key)  
			print(reader.get_tensor(key)) # Remove this is you want to print only variable names  