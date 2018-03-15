#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:38:36 2017

@author: lyj
"""


'''
DnCNN
paper: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising

a tensorflow version of the network DnCNN
just for personal exercise

author: momogary
'''

import tensorflow as tf 
import numpy as np 
import math, os
from glob import glob
from ops import *
from utils import *
from six.moves import xrange
import time
import matplotlib.pyplot as plt
class DnCNN(object):
	def __init__(self, sess, image_size=40, batch_size=128,
					output_size=40, input_c_dim=1, output_c_dim=1, 
					sigma=25, clip_b=0.025, lr=0.01, epoch=50,
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
		self.reuse = False
		self.build_model(self.reuse)


	def build_model(self,reuse):
		# input : [batchsize, image_size, image_size, channel]
		self.X = tf.placeholder(tf.float32, \
					[None, self.image_size, self.image_size, self.input_c_dim], \
					name='noisy_image')
		self.X_ = tf.placeholder(tf.float32, \
					[None, self.image_size, self.image_size, self.input_c_dim], \
					name='clean_image')
		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		# layer 1
		with tf.variable_scope('conv1', reuse=reuse):
			layer_1_output = self.layer(self.X, [3, 3, self.input_c_dim, 64], useBN=False)
		# layer 2 to 16
		with tf.variable_scope('conv2', reuse=reuse):
			layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
		with tf.variable_scope('conv3', reuse=reuse):
			layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64])
		with tf.variable_scope('conv4', reuse=reuse):
			layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64])
		with tf.variable_scope('conv5', reuse=reuse):
			layer_5_output = self.layer(layer_4_output, [3, 3, 64, 64])
		with tf.variable_scope('conv6', reuse=reuse):
			layer_6_output = self.layer(layer_5_output, [3, 3, 64, 64])
		with tf.variable_scope('conv7', reuse=reuse):
			layer_7_output = self.layer(layer_6_output, [3, 3, 64, 64])
		with tf.variable_scope('conv8', reuse=reuse):
			layer_8_output = self.layer(layer_7_output, [3, 3, 64, 64])
		with tf.variable_scope('conv9', reuse=reuse):
			layer_9_output = self.layer(layer_8_output, [3, 3, 64, 64])
		with tf.variable_scope('conv10', reuse=reuse):
			layer_10_output = self.layer(layer_9_output, [3, 3, 64, 64])
		with tf.variable_scope('conv11', reuse=reuse):
			layer_11_output = self.layer(layer_10_output, [3, 3, 64, 64])
		with tf.variable_scope('conv12', reuse=reuse):
			layer_12_output = self.layer(layer_11_output, [3, 3, 64, 64])
		with tf.variable_scope('conv13', reuse=reuse):
			layer_13_output = self.layer(layer_12_output, [3, 3, 64, 64])
		with tf.variable_scope('conv14', reuse=reuse):
			layer_14_output = self.layer(layer_13_output, [3, 3, 64, 64])
		with tf.variable_scope('conv15', reuse=reuse):
			layer_15_output = self.layer(layer_14_output, [3, 3, 64, 64])
		with tf.variable_scope('conv16', reuse=reuse):
			layer_16_output = self.layer(layer_15_output, [3, 3, 64, 64])

		# layer 17
		with tf.variable_scope('conv17', reuse=reuse):
			self.Y = self.conv_layer(layer_16_output, [3, 3, 64, self.output_c_dim],  b_init = 0.0, stridemode=[1,1,1,1])
		#print(tf.get_variable("conv1/weights",[1])) 
		# L2 loss
		self.Y_ = self.X - self.X_ # noisy image - clean image
		self.loss = (1.0 / self.batch_size) * tf.nn.l2_loss(self.Y - self.Y_)
		self.l2 = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		self.loss = self.loss + self.l2
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='AdamOptimizer')
		self.train_step = optimizer.minimize(self.loss)
		# create this init op after all variables specified
		self.init = tf.global_variables_initializer() 
		self.saver = tf.train.Saver(max_to_keep=100000)
		print("[*] Initialize model successfully...")

	def conv_layer(self, inputdata, weightshape, b_init, stridemode):
		# weights
		W = tf.get_variable('weights', regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0001) ,shape = weightshape, initializer = \
							tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
#		self.W_params = self.W_params.append([W])
#		print(W.name)
		b = tf.get_variable('biases', [1, weightshape[-1]], initializer = \
							tf.constant_initializer(b_init))
#		self.b_params = self.b_params.append(b)
#		print(b.name)
		# convolutional layer
		return tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME") + b # SAME with zero padding

	def bn_layer(self, logits, output_dim, b_init = 0.0):
		alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer = \
								tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)))
		beta = tf.get_variable('bn_beta', [1, output_dim], initializer = \
								tf.constant_initializer(b_init))
		return batch_normalization(logits, alpha, beta, isCovNet = True)

	def per_layer(self, inputdata, input_c_dim,useBN):
		with tf.variable_scope('1x1', reuse=reuse):
			output_1x1 = self.layer(inputdata, [1, 1, input_c_dim, 16], useBN=False)     
		with tf.variable_scope('3x3', reuse=reuse):
			output_3x3 = self.layer(inputdata, [3, 3, input_c_dim, 64], useBN=False)   
		with tf.variable_scope('5x5', reuse=reuse):
			output_5x5 = self.layer(inputdata, [5, 5, input_c_dim, 32], useBN=False)   
		with tf.variable_scope('7x7', reuse=reuse):
			output_7x7 = self.layer(inputdata, [7, 7, input_c_dim, 16], useBN=False)
		concat_output=tf.concat([output_1x1,output_3x3,output_5x5,output_7x7],axis=3)
		with tf.variable_scope('reduce', reuse=reuse):
			output = self.layer(concat_output, [1, 1, 128, 64], useBN=False)
		return output
  
	def layer(self, inputdata, filter_shape, b_init = 0.0, stridemode=[1,1,1,1], useBN = True):
		logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode)
		if useBN:
			output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1]))
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
		f = file('/home/lyj/tensorflow_code/DnCNN-tensorflow-master/dataset_2.txt','w+')
		self.sess.run(self.init)

		# get data
		test_files = glob('./test/*.png'.format(self.testset))
		#test_files = './test/*.png'
		#print(test_files)
		test_data = load_images(test_files) # list of array of different size, 4-D
		data = load_data(filepath='./img_clean_pats.npy')
		noise_data = add_noise(data, self.sigma, self.sess)
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
		costs = []
		for epoch in xrange(self.epoch):
			for batch_id in xrange(numBatch):
				batch_images = data[batch_id*self.batch_size:(batch_id+1)*self.batch_size, :, :, :]
				train_images = noise_data[batch_id*self.batch_size:(batch_id+1)*self.batch_size, :, :, :]
#				if epoch < 20:
#					_, loss ,l2 = self.sess.run([self.train_step, self.loss , self.l2], \
#						feed_dict={self.X:add_noise(data[batch_id*self.batch_size:(batch_id+1)*self.batch_size, :, :, :], self.sigma, self.sess), self.X_:data[batch_id*self.batch_size:(batch_id+1)*self.batch_size, :, :, :],self.learning_rate: 0.01})
				if epoch < 20:
					_, loss ,l2 = self.sess.run([self.train_step, self.loss , self.l2], \
						feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.01})
				elif epoch >= 20 and epoch < 40:
					_, loss ,l2 = self.sess.run([self.train_step, self.loss , self.l2], \
						feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.001})
				elif epoch >= 40:
					_, loss ,l2 = self.sess.run([self.train_step, self.loss , self.l2], \
						feed_dict={self.X:train_images, self.X_:batch_images,self.learning_rate: 0.0001})
#				batch_images = []
#				train_images = []
				costs.append(loss)
				last_time = this_time
				this_time = time.time()
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, loss_l2: %.6f" \
					% (epoch + 1, batch_id + 1, numBatch,
						time.time() - last_time, self.mn(costs), l2))
				#print(test_files)
				#print(test_data)
				counter += 1
				f.write("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, loss_l2: %.6f" \
					% (epoch + 1, batch_id + 1, numBatch,
						time.time() - last_time, self.mn(costs), l2))
				f.write("\n")

#				if np.mod(counter, self.eval_every_iter) == 0:
#					self.evaluate(epoch, counter, test_data,test_files)
				# save the model
				if np.mod(counter, self.save_every_iter) == 0:
					self.save(counter)
		print("[*] Finish training.")
		f.close()

  
	def save(self,counter):
		model_name = "DnCNN.model"
		model_dir = "dataset_%s_%s_%s" % (self.trainset, \
									self.batch_size, self.image_size)
		checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		print("[*] Saving model...")
		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=counter)

	def sampler(self,X_test):
		# set reuse flag to True
		#tf.get_variable_scope().reuse_variables()

		# layer 1 (adpat to the input image)
		with tf.variable_scope('conv1', reuse=True):
			layer_1_output = self.layer(X_test, [3, 3, self.input_c_dim, 64], useBN=False)

		# layer 2 to 16
		with tf.variable_scope('conv2', reuse=True):
			layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
		with tf.variable_scope('conv3', reuse=True):
			layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64])
		with tf.variable_scope('conv4', reuse=True):
			layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64])
		with tf.variable_scope('conv5', reuse=True):
			layer_5_output = self.layer(layer_4_output, [3, 3, 64, 64])
		with tf.variable_scope('conv6', reuse=True):
			layer_6_output = self.layer(layer_5_output, [3, 3, 64, 64])
		with tf.variable_scope('conv7', reuse=True):
			layer_7_output = self.layer(layer_6_output, [3, 3, 64, 64])
		with tf.variable_scope('conv8', reuse=True):
			layer_8_output = self.layer(layer_7_output, [3, 3, 64, 64])
		with tf.variable_scope('conv9', reuse=True):
			layer_9_output = self.layer(layer_8_output, [3, 3, 64, 64])
		with tf.variable_scope('conv10', reuse=True):
			layer_10_output = self.layer(layer_9_output, [3, 3, 64, 64])
		with tf.variable_scope('conv11', reuse=True):
			layer_11_output = self.layer(layer_10_output, [3, 3, 64, 64])
		with tf.variable_scope('conv12', reuse=True):
			layer_12_output = self.layer(layer_11_output, [3, 3, 64, 64])
		with tf.variable_scope('conv13', reuse=True):
			layer_13_output = self.layer(layer_12_output, [3, 3, 64, 64])
		with tf.variable_scope('conv14', reuse=True):
			layer_14_output = self.layer(layer_13_output, [3, 3, 64, 64])
		with tf.variable_scope('conv15', reuse=True):
			layer_15_output = self.layer(layer_14_output, [3, 3, 64, 64])
		with tf.variable_scope('conv16', reuse=True):
			layer_16_output = self.layer(layer_15_output, [3, 3, 64, 64])

		# layer 17
		with tf.variable_scope('conv17', reuse=True):
			self.Y_test = self.conv_layer(layer_16_output, [3, 3, 64, self.output_c_dim],  b_init = 0.0, stridemode=[1,1,1,1])
		return self.Y_test
                       
	def load(self, checkpoint_dir):
		'''Load checkpoint file'''
		print("[*] Reading checkpoint...")

		model_dir = "%s_%s_%s" % (self.trainset, self.batch_size, self.image_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		print(checkpoint_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False


	def test(self):
		"""Test DnCNN"""
		# init variables
		tf.initialize_all_variables().run()
		test_files = glob('./test/*.png'.format(self.testset))
#		test_files = glob(os.path.join(self.test_save_dir, '{}/*.png'.format(self.testset)))
		print(test_files)
		# load testing input
		X_test = tf.placeholder(tf.float32, \
					[1,None,None,1], name='noisy_image_test')
		predicted_noise = self.sampler(X_test)
		print("[*] Loading test images ...")
		test_data = load_images(test_files) # list of array of different size
		for order in range(1,60): 		
        		model_path = './checkpoint/dataset_BSD400_128_40/DnCNN.model-%d'%(order*500)
        		saver = tf.train.Saver()
        		saver.restore(self.sess, model_path)
        		psnr_sum = 0
        		for idx in xrange(len(test_files)):
        			noisy_image = add_noise(1 / 255.0 * test_data[idx], self.sigma, self.sess)  # ndarray
        			predicted_noise_ = self.sess.run([predicted_noise],feed_dict={X_test : noisy_image})

        			output_clean_image = (noisy_image - predicted_noise_) * 255
 			# calculate PSNR
        			psnr = cal_psnr(test_data[idx], output_clean_image)
        			psnr_sum += psnr
#			save_images(test_data[idx], noisy_image * 255, output_clean_image, \
#						os.path.join(self.test_save_dir, '%s/test%d_%d_%d.png' % (self.testset, idx, epoch, counter)))
        		avg_psnr = psnr_sum / len(test_files)
        		print("--- Average PSNR %.2f ---" % avg_psnr)

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
