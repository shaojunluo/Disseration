#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:59:36 2018

@author: shaojun
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import random as rdm

#from matplotlib import pyplot as plt

# aggregate data
def group_class(Y,n):
    Y_out = {}
    for key,value in Y.iteritems():
            Y_out[key] = int(int(value) > n)
    return Y_out

# load training set:
print('Loading Test Data')

#%Calculate accuracy for test images
#X_test = np.load('/media/shaojun/bottleneck_3/BIRADS/x_test.npy').item()
L_test_BIRADS = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/l_test.npy').item()
X_test_origin = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/x_test.npy').item()
L_test_PATH = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/p_test.npy').item()
# test on training set
#L_test_BIRADS = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/l_train.npy').item()
#X_test_origin = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/x_train.npy').item()
#L_test_PATH = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/p_train.npy').item()
#%% testset for selection network 
L_test = {}
X_test = {}
B_test = {}

#X_test = X_test_origin
#L_test = group_class(L_test_BIRADS,3)
#B_test = L_test_BIRADS

#L_test = L_test_BIRADS
for key in L_test_BIRADS:
    if int(L_test_BIRADS[key]) >= 0:
        X_test[key] = X_test_origin[key]
        L_test[key] = L_test_PATH[key]
        B_test[key] = L_test_BIRADS[key]
        
# build class_map
def build_class_map(class_list):
    out_dict = {}
    n = 0
    for key in class_list:
        # if class is aggregated
        if type(key) is list:
            # numbering aggregated class
            for k in key:
                out_dict[k] = n
        # non-aggregated class
        else:
            out_dict[key] = n
        n += 1
    return out_dict

# pop short scans
def filter_scans(window_size):
    num_rmv = 0
    for key in X_test.keys():
        if X_test[key].shape[0]<window_size:
            X_test.pop(key)
            L_test.pop(key)
            num_rmv += 1
    
    print(str(num_rmv)+' scans removed')
    
#%%        
filter_scans(3)
meta_step = 470
# SHAPE OF NETWORK
num_input = 2048 # feature data input size
window_size = 3 # size of window batch selection
#build class_map first
class_list = [0,1]
class_map = build_class_map(class_list) # map the classes

# generator for split dictionary
def chunks(seq, size = 1000):
    for i in xrange(0, len(seq), size):
        yield seq[i:i+size]

# reshape to feed rnn
def reshape_to_feed(X, L, samples,window_size, 
                    slice_map = None,class_map = None, class_num = None):
    # if input is for SLCT inferring
    if slice_map is None:
        size = X[samples].shape[0] - window_size + 1
        x = np.zeros((size,window_size,num_input))
        for i in range(size):
            x[i,:,:] = X[samples][i : i + window_size, :]
        y = L[samples]
        key_list = samples
    # Else for DGNS
    else:   
        size = len(samples)
        # create x,y batches
        x = np.zeros((size,window_size,num_input))
        if class_num is None:
            y = np.zeros((size,len(class_map)))
        else:
            y = np.zeros((size,class_num))
        key_list = []
        i = 0
        for key in samples:
            # pick up the predicted window
            x[i,:,:] = X[key][slice_map[key] : slice_map[key] + window_size, :]
            # assigning classes
            if class_map is None:
                y[i,L[key]] = 1
            else:
                y[i,class_map[L[key]]] = 1
            # attach keys
            key_list.append(key)
            i+=1
    return x, y, key_list

# expand the slices for UA network
def expand_slices(x,window_size):
    size = x.shape[0]-window_size+1
    out_x = np.zeros((size,window_size,num_input))
    for i in range(size):
        out_x[i,:,:] = x[i : i+ window_size,:]
    return out_x

#meta_step = 250
#tf.reset_default_graph()
print("Loading Models")
with tf.Session() as sess:
    imported_meta = tf.train.import_meta_graph('models/RRNN_v3/T1/RRNN_3-'+str(meta_step)+'.meta')
    imported_meta.restore(sess,'models/RRNN_v3/T1/RRNN_3-'+str(meta_step))
    
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    DGNS_label_pred = graph.get_tensor_by_name("DGNS_label_pred:0")
    DGNS_acc = graph.get_tensor_by_name("DGNS_acc:0")
    UA_argmax = graph.get_tensor_by_name("UA_argmax:0")
    # filter the result by thresholding
    DGNS_logits = graph.get_tensor_by_name('DGNS_logits:0')
    
    
    #% test all
    print('Test on total dataset')

    # parse the most likely slice
    slice_map = {}
    for key in L_test.keys():
        # reshape to feed, x,y
        x = expand_slices(X_test[key],window_size)
        # label the prediction to check weather it is correct
        slice_map[key] = int(sess.run(UA_argmax,feed_dict={X: x}))
    # Calculate the prediction result
    y_logits = []
    y_t_tot = []
    y_p_tot = []
    y_b_tot = []
    pred_data = pd.DataFrame(columns = ['logits_0','logits_1','True_Label','Pred_Label','BIRADS'])
    for samples in chunks(X_test.keys()):
        x,y,key_list = reshape_to_feed(X_test,L_test,samples, window_size,
                                       slice_map = slice_map,class_map = class_map)
        # label the prediction to check weather it is correct
        pred_label,pred_logits = sess.run([DGNS_label_pred,DGNS_logits],feed_dict={X: x,Y: y})

        temp_data = pd.DataFrame()
        temp_data['keys'] = key_list
        temp_data['logits_0'] = pred_logits[:,0]
        temp_data['logits_1'] = pred_logits[:,1]
        temp_data['Pred_Label'] = pred_label
        temp_data['True_Label'] = np.argmax(y,axis = 1)
        temp_data['BIRADS'] = [B_test[key] for key in key_list]

        pred_data = pred_data.append(temp_data)
    pred_data.set_index('keys')

pred_data.to_csv('results/T1_test.csv',index = False)
#%% link the data to Pathology result

plt.figure()
ax = plt.subplot()
for i in range(0,2):
    #n = 220 + i
    if i == 0:
        pred_data = pd.read_csv('results/T1_test.csv')
    else:
        pred_data = pd.read_csv('results/Task_1_test.csv')
    y_t_tot = np.array(pred_data['True_Label'],dtype = float)
    y_p_tot = np.array(pred_data['Pred_Label'],dtype = float)
    y_logits = np.array(pred_data['logits_1'] -  pred_data['logits_0'],dtype = float)
    
    acc_tot = 0
    fpr = 0
    tpr = 0
    auc = 0
    k = 1000
    for j in range(k):

        # balanced sample
        idx_ma = np.where(y_t_tot > 0)[0]
        idx_be = np.where(y_t_tot < 1)[0]
        idx = np.concatenate((rdm.sample(idx_be,len(idx_ma)), idx_ma))
    
        y_t = y_t_tot[idx]
        y_p = y_p_tot[idx]
        y_lg = y_logits[idx]
                    
        conf_mat_tot = confusion_matrix(y_t,y_p)
        acc_tot += float(np.sum(conf_mat_tot.diagonal()))/np.sum(conf_mat_tot)
        #print('Confusion Matrix for Task '+str(i))
        #print(conf_mat_tot)
        
        #np.save('conf_mat_tot.npy',conf_mat_tot)
    
        fpr, tpr, thres = roc_curve(y_t, y_lg)
        auc += 0.5*np.dot(fpr[1:]-fpr[:-1],tpr[1:] + tpr[:-1] )
    print('Test Accuracy: {:.3f}'.format(acc_tot/k))
    print('Area Under the Curve: {:.3f}'.format(auc/k))
    ax.plot(fpr,tpr) #color='darkorange')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    if i!=3:
        ax.plot(0.512, 1.0,'+')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.axis('equal')

#%% ROC curve for different birads

new_data = y