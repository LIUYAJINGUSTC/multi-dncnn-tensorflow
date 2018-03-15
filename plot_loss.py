#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:06:47 2017

@author: lyj
"""
import numpy as np
import matplotlib.pyplot as plt

txtpath="loss.txt"

fp=open(txtpath)
training_g_error=[]
training_d_error=[]
training_mse_error=[]
training_mean_error=[]
testing_g_error=[]
testing_d_error=[]
testing_mse_error=[]
for lines in fp.readlines():
    lists = lines.split(' ') #分割出文件与文件扩展名
    
    if lists[1] and lists[1]=='loss:':
        if lists[2]:
            training_g_error.append(np.float(lists[2]))
        
#        training_d_error.append(np.float(lists[2]))
#        training_mse_error.append(np.float(lists[3]))
#    if lists[0]=='test mean errors : ':
#        testing_g_error.append(np.float(lists[1]))
#        testing_d_error.append(np.float(lists[2]))
#        testing_mse_error.append(np.float(lists[3]))
    #lines=lines.replace("\n","").split(",")
    #arr.append(lines)
training_g_error=np.array(training_g_error)
fp.close()
#plt.plot(training_mse_error)
#plt.plot(training_g_error)

plt.figure()
plt.title('training_d_error') 
#plt.plot(training_d_error[10000:70000])
plt.plot(training_d_error)
plt.figure()
#plt.title('training_g_error') 
plt.plot(training_g_error[1000:70000])
#plt.plot(training_g_error)
#plt.figure()
##plt.title('training_mse_error') 
##plt.plot(training_mse_error[10000:70000])
#plt.plot(training_mse_error)
#plt.figure()
##plt.title('testing_mse_error') 
##plt.plot(testing_mse_error)
##plt.figure()
##plt.title('testing_g_error') 
##plt.plot(testing_g_error)
##plt.figure()
##plt.title('testing_d_error') 
#plt.plot(testing_d_error)
#plt.plot(training_mse_error)
#plt.plot(training_mse_error)
