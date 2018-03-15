import numpy as np 
import os, sys
from PIL import Image
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import pdb
SEED = 66478
def data_augmentation(image, mode):
	if mode == 0:
		# original
		return image
	elif mode == 1:
		# flip up and down
		return np.flipud(image)
	elif mode == 2:
		# rotate counterwise 90 degree
		return np.rot90(image)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		image = np.rot90(image)
		return np.flipud(image)
	elif mode == 4:
		# rotate 180 degree
		return np.rot90(image, k = 2)
	elif mode == 5:
		# rotate 180 degree and flip
		image = np.rot90(image, k = 2)
		return np.flipud(image)
	elif mode == 6:
		# rotate 270 degree
		return np.rot90(image, k = 3)
	elif mode == 7:
		# rotate 270 degree and flip
		image = np.rot90(image, k = 3)
		return np.flipud(image)

def load_data(filepath='./image_clean_pat.npy'):
	assert '.npy' in filepath
	if not os.path.exists(filepath):
		print("[!] Data file not exists")
		sys.exit(1)

	print("[*] Loading data...")
	data = np.load(filepath)
	np.random.shuffle(data)
	print("[*] Load successfully...")
	return data

def add_noise(data, sigma, sess):
	noise = sigma / 255.0 * sess.run(tf.random_normal(data.shape,seed=0))
#	noise = sigma / 255.0 * sess.run(tf.random_normal(data.shape))
	return (data + noise)

def load_images(filelist):
	data = []
	for file in filelist:
		im = Image.open(file).convert('L')
		data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
#		plt.figure()
#		image = np.array(im).reshape(1, im.size[1], im.size[0], 1)
# 		image = image[0,:,:,0]
# 		plt.imshow(image, cmap ='gray')
#		pdb.set_trace()
	return data

def save_images(ground_truth, noisy_image, clean_image, filepath):
	_, im_h, im_w = noisy_image.shape
	ground_truth = ground_truth.reshape((im_h, im_w))
	noisy_image = noisy_image.reshape((im_h, im_w))
	clean_image = clean_image.reshape((im_h, im_w))
	cat_image = np.column_stack((noisy_image, clean_image))
	cat_image = np.column_stack((ground_truth, cat_image))
	im = Image.fromarray(clean_image.astype('uint8')).convert('L')
	im.save(filepath, 'png')

def cal_psnr(X_pred,target):
    X_pred = X_pred.astype('Float32')
    target = target.astype('Float32')
    sub_val=X_pred-target
    squa = np.multiply(sub_val,sub_val)
    squa_mean=squa.mean()
    a=math.sqrt(squa_mean)
    b=255/a
    c=math.log(b,10)
    psnr=20*c
    return psnr    
    
#    mse = (np.abs(X_pred - target) ** 2).sum() / (target.shape[1] * target.shape[2])
#    psnr = 10 * np.log10(255 * 255 / mse)
#    return psnr