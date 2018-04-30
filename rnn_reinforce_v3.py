#!/usr/bin/env python2
# -*- coding: utf-8 -*-

""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""


from __future__ import print_function

import tensorflow as tf
import numpy as np
import random as rdm
from tensorflow.contrib import rnn
#from matplotlib import pyplot as plt

# aggregate data
def group_class(Y,n):
    Y_out = {}
    for key,value in Y.iteritems():
            Y_out[key] = int(int(value) > n)
    return Y_out

# load training set:
print('Loading Training Data')
#X_train = np.load('/media/shaojun/bottleneck_3/BIRADS/x_train.npy').item()
L_train_BIRADS = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/l_train.npy').item()
X_train_origin = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/x_train.npy').item()
L_train_PATH = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/p_train.npy').item()

#%Calculate accuracy for test images
#X_test = np.load('/media/shaojun/bottleneck_3/BIRADS/x_test.npy').item()
L_test_BIRADS = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/l_test.npy').item()
X_test_origin = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/x_test.npy').item()
L_test_PATH = np.load('/media/shaojun/bottleneck_3/pathology/03_21_t1/p_test.npy').item()

# taining set for selection network 
R_train = {}
L_train = {}
X_train = {}
L_test = {}
X_test = {}
#% Dataset and functions
# binary split with >=4 and <4
#X_train = X_train_origin
#L_train = group_class(L_train_BIRADS,3)
for key in L_train_BIRADS:
    if int(L_train_BIRADS[key]) >= 0:
        X_train[key] = X_train_origin[key]
        L_train[key] = L_train_PATH[key]
# binary split with >=4 and <4
#X_test = X_test_origin
#L_test = group_class(L_test_BIRADS,3)
for key in L_test_BIRADS:
    if int(L_test_BIRADS[key]) >= 0:
        X_test[key] = X_test_origin[key]
        L_test[key] = L_test_PATH[key]

#%% Parameter and Network construction
# the overall training loop
#meta_steps = 200
display_step = 10

meta_step_1 = 100
meta_step_2 = 500
alpha = 3.0
#% Training Parameters for SLCT network
SLCT_steps = 500
SLCT_learning_rate = 1e-3
#% Training Parameters for DGNS network
DGNS_steps = 500
DGNS_steps_final = 1000
DGNS_learning_rate = 1e-3
#% Training Parameters for UA network
UA_steps = 500
UA_learning_rate = 1e-4

# SHAPE OF NETWORK
num_input = 2048 # feature data input size

# Selection Network Parameters
SLCT_hidden = 50 # if no hidden layer, then same as input
SLCT_classes = 2
# Diagnose Network Parameters
DGNS_hidden = 50 # hidden layer num of features
class_list = [0,1]
DGNS_classes = len(class_list) # BIRAD total classes specified by class_list
window_size = 3 # size of window batch selection

UA_hidden = 50
# feed batch size
DGNS_batch_size = 20 # DGNS network batch size for EACH classes
SLCT_batch_size = 20 # SLCT network batch size for EACH classes
UA_batch_size = 20 # UA network batch size
validate_batch = 100 # batch for validation

start_step = 0

#% Necessary functions
# build one hot class map
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

#% define length of sequence
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

# pre classify the sample
def pre_classify(Y,class_list = None):
    if class_list is None:
        class_list = np.unique(Y.values())
    classified_keys = []
    for l in class_list:
        if type(l) is list:
            classified_keys.append([key for key,value in Y.iteritems() if int(value) in l])
        else:
            classified_keys.append([key for key,value in Y.iteritems() if value == l])
    return classified_keys

# pre processing data
def balanced_sampling(Y,k,class_list = None,pre_list = None):
    # balanced subsampling
    sub_keys = []
    # the pre-list is existing
    if pre_list is None:
        classified_keys = pre_classify(Y,class_list = class_list)
    else:
        classified_keys = pre_list
    # balanced sample,each k
    for sample_keys in classified_keys:
        sub_keys.extend(rdm.sample(sample_keys,k))
    # random shuffle
    rdm.shuffle(sub_keys)
    return sub_keys,classified_keys

# pop short scans
def filter_scans(window_size):
    num_rmv = 0
    for key in X_train.keys():
        if X_train[key].shape[0]<window_size:
            X_train.pop(key)
            L_train.pop(key)
            num_rmv += 1
    for key in X_test.keys():
        if X_test[key].shape[0]<window_size:
            X_test.pop(key)
            L_test.pop(key)
            num_rmv += 1
    
    print(str(num_rmv)+' scans removed')
    
# find max sum window
def max_sum_window(x,window_size):
    x_sum = np.sum(x, axis = 1)
    # pick up the TOP max sum
    conv_sum = np.convolve(x_sum, np.ones(window_size), mode = 'valid')
    # pick up the incedices max sum
    max_sum_idx = np.argmax(conv_sum)
    return max_sum_idx

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

# create slice pool:
def random_slice_pick(x,s,batch_size):
    out_x = np.zeros((batch_size,window_size,num_input))
    out_s = np.zeros((batch_size,1))
    key_samples = rdm.sample(s.keys(),batch_size)
    i = 0
    for key in key_samples:
        m = rdm.randint(0,x[key].shape[0]-window_size)
        out_x[i,:,:] = x[key][m : m + window_size, :]
        out_s[i] = np.sum(s[key][m : m + window_size])
        i+= 1
    return out_x, out_s

# expand the slices for UA network
def expand_slices(x,window_size):
    size = x.shape[0]-window_size+1
    out_x = np.zeros((size,window_size,num_input))
    for i in range(size):
        out_x[i,:,:] = x[i : i+ window_size,:]
    return out_x

# update score list
def update_score(key,label,alpha,step,damp =0.1):
    m = slice_map[key]
    score_map[key]*= 1-damp
    # score of R. label is 1 when good prediction, 0 wrong prediction
    R = L_train[key]*label + 0.5*(1 - L_train[key])*(1 - label)
    score_map[key]+= R*np.exp(-(np.arange(score_map[key].shape[0])-m)**2/(2*alpha**2))

#build class_map first
class_map = build_class_map(class_list) # map the classes

# tensorflow settings
tf.reset_default_graph() 

# tf Graph input
X = tf.placeholder("float", [None,window_size,num_input],name = 'X')
Y = tf.placeholder("float", [None,DGNS_classes],name = 'Y')
Y_S = tf.placeholder("float", [None,SLCT_classes],name = 'Y_S')
SC = tf.placeholder("float", [None,1],name = 'SC')

# Define weights
weights = {
    'SLCT_out': tf.Variable(tf.random_normal([2*SLCT_hidden, SLCT_classes])),
    'DGNS_out': tf.Variable(tf.random_normal([2*DGNS_hidden, DGNS_classes])),
    'UA_w': tf.Variable(tf.random_normal([2*UA_hidden, 1])),
}
biases = {
    'SLCT_out': tf.Variable(tf.random_normal([SLCT_classes])),
    'DGNS_out': tf.Variable(tf.random_normal([DGNS_classes])),
    'UA_b': tf.Variable(tf.random_normal([1]))
}

# last relevant sequence
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

# selection network, LSTM RNN
def SLCT_NET(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (time_steps,ninput)
    # Required shape: 'timesteps' tensors list of shape (n_input)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(SLCT_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(SLCT_hidden, forget_bias=1.0)
    # Get lstm cell output
    #outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
    #                                   sequence_length=length(x))
    # Get lstm cell output
    (output_fw,output_bw),state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32,sequence_length=length(x))
    output = tf.concat([output_fw, output_bw], axis=2)
    # Linear activation, using rnn inner loop last output
    last = last_relevant(output,length(x))
    return tf.matmul(last, weights['SLCT_out']) + biases['SLCT_out']

with tf.variable_scope('SLCT') as vs:
    SLCT_logits = SLCT_NET(X, weights, biases)
    # Retrieve just the LSTM variables.
    SLCT_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
    
# save SLCT_logits    
SLCT_logits = tf.add(SLCT_logits, 0.0, name ='SLCT_logits')
SLCT_argmax = tf.argmax(SLCT_logits[:,1]-SLCT_logits[:,0],name = 'SLCT_argmax')
#%
SLCT_pred = tf.nn.softmax(SLCT_logits)
SLCT_label_pred = tf.argmax(SLCT_pred, 1, name = 'SLCT_label_pred')
# Define loss and optimizer
SLCT_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            logits = SLCT_logits, labels = Y_S),name = 'SLCT_loss')
SLCT_optimizer = tf.train.AdamOptimizer(learning_rate = SLCT_learning_rate)
SLCT_train = SLCT_optimizer.minimize(SLCT_loss,name = 'SLCT_train')

# Evaluate model (with test logits, for dropout to be disabled)
SLCT_correct_pred = tf.equal(tf.argmax(SLCT_pred, 1), tf.argmax(Y_S, 1))
SLCT_acc = tf.reduce_mean(tf.cast(SLCT_correct_pred, tf.float32),name = 'SLCT_acc')

# diagnose network, LSTM RNN
def DGNS_NET(x, weights, biases):
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell_2 = rnn.BasicLSTMCell(DGNS_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell_2 = rnn.BasicLSTMCell(DGNS_hidden, forget_bias=1.0)
    # Get lstm cell output
    #outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
    #                                   sequence_length=length(x))
    # Get lstm cell output
    (output_fw_2,output_bw_2),state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, x,
                                              dtype=tf.float32,sequence_length=length(x))
    output_2 = tf.concat([output_fw_2, output_bw_2], axis=2)
    # Linear activation, using rnn inner loop last output
    last_2 = last_relevant(output_2,length(x))
    return tf.add(tf.matmul(last_2, weights['DGNS_out']), biases['DGNS_out'])

#initializer of DGNS network
with tf.variable_scope('DGNS') as vs:
    DGNS_logits = DGNS_NET(X, weights, biases)
    # Retrieve just the LSTM variables.
    DGNS_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]

DGNS_pred = tf.nn.softmax(DGNS_logits)
DGNS_label_pred = tf.argmax(DGNS_pred, 1,name = 'DGNS_label_pred')
DGNS_logits = tf.add(DGNS_logits, 0.0, name ='DGNS_logits')
# Define loss and optimizer
DGNS_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=DGNS_logits, labels=Y))
DGNS_optimizer = tf.train.AdamOptimizer(learning_rate = DGNS_learning_rate)
DGNS_train = DGNS_optimizer.minimize(DGNS_loss)

# Evaluate model (with test logits, for dropout to be disabled)
DGNS_correct_pred = tf.equal(tf.argmax(DGNS_pred, 1), tf.argmax(Y, 1))
DGNS_acc = tf.reduce_mean(tf.cast(DGNS_correct_pred, tf.float32),name = 'DGNS_acc')

# Universal Approximator
def SLCT_UA_NET(x, weights,biases):
    # first go through SLCT net
    # Forward direction cell
    lstm_fw_cell_3 = rnn.BasicLSTMCell(UA_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell_3 = rnn.BasicLSTMCell(UA_hidden, forget_bias=1.0)
    # Get lstm cell output
    (output_fw_3,output_bw_3),state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_3, lstm_bw_cell_3, x,
                                              dtype=tf.float32,sequence_length=length(x))
    output_3 = tf.concat([output_fw_3, output_bw_3], axis=2)
    # Linear activation, using rnn inner loop last output
    last_3 = last_relevant(output_3,length(x))
    # then feed univeral 
    return tf.matmul(last_3, weights['UA_w']) + biases['UA_b'] 
# universal apprximater
with tf.variable_scope('SLCT_UA_NET') as vs:
    UA_logits = SLCT_UA_NET(X,weights, biases)
    UA_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]
UA_logits = tf.add(UA_logits, 0.0, name ='UA_logits')
UA_argmax = tf.argmax(UA_logits,0,name = 'UA_argmax')
UA_loss = tf.reduce_mean(tf.square(UA_logits - SC),name = 'UA_loss')
UA_optimizer = tf.train.AdamOptimizer(learning_rate = UA_learning_rate)
UA_train = UA_optimizer.minimize(UA_loss,name = 'UA_train')
UA_tot_err = tf.reduce_sum(tf.square(SC - tf.reduce_mean(SC)))
UA_uex_err = tf.reduce_sum(tf.square(UA_logits - SC))
UA_R_squared = tf.subtract(1.0, tf.div(UA_uex_err,UA_tot_err),name = 'UA_R_squared')

#%% Start training
filter_scans(window_size)
    
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    
    # initializing slice_map
    print('initializing slice selection')
    # restoring graph session
    if start_step > 0:
        imported_meta = tf.train.Saver()
        imported_meta.restore(sess,'models/RRNN_v3/T1/final_'+str(window_size)+'-'+str(start_step))
        graph = tf.get_default_graph()
        # read the weights and operations of preivious network
        SLCT_loss = graph.get_tensor_by_name("SLCT_loss:0")
        # continue training
        SLCT_train = SLCT_optimizer.minimize(SLCT_loss)
        slice_map = np.load('models/RRNN_v3/T1/final_slice_'+str(window_size)+'-'+str(start_step)+'.npy').item()
        score_map = np.load('models/RRNN_v3/T1/final_score_'+str(window_size)+'-'+str(start_step)+'.npy').item()
        # record lists
        loss_list = list(np.load('models/RRNN_v3/T1/loss_list_'+str(window_size)+'-'+str(start_step)+'.npy'))
        acc_list = list(np.load('models/RRNN_v3/T1/acc_list_'+str(window_size)+'-'+str(start_step)+'.npy'))
        change_list = list(np.load('models/RRNN_v3/T1/change_list_'+str(window_size)+'-'+str(start_step)+'.npy'))
        R_squared_list = list(np.load('models/RRNN_v3/T1/R_squared_list_'+str(window_size)+'-'+str(start_step)+'.npy'))
    else:
    # if nothing to restore, calculate the max)sum
        #% start training loop
        loss_list = []
        acc_list = []
        change_list = []
        #print('Use Pretrained slice_map')
        #slice_map = np.load('models/RRNN_v1/final_slice_3-300.npy').item()

        slice_map = {}
        for key in X_train.keys():
            slice_map[key] = max_sum_window(X_train[key],window_size)
        # initialize score map
        score_map = {}
        for key in X_train.keys():
            # score map. the first element is the number of good prediction
            # second is the number of choosed
            score_map[key] = np.zeros(X_train[key].shape[0]).astype(float)
    #loop parameter
    meta_step = start_step
    # define initialization for DGNS network
    DGNS_var_list = [weights['DGNS_out'], biases['DGNS_out']]+DGNS_variables
    DGNS_init = tf.variables_initializer(DGNS_var_list, name = 'DGNS_init')
    UA_var_list = [weights['UA_w'], biases['UA_b']]+UA_variables
    UA_init = tf.variables_initializer(UA_var_list, name = 'UA_init')
    
    print('Training')

    change = 1
    acc = 0.0
    
    # set up sample list
    L_list = pre_classify(L_train,class_list = class_list)
    # set up validation list
    V_list = pre_classify(L_test,class_list = class_list)
    
    while (change > 0.05 and meta_step < start_step + meta_step_1):
        # initialize DGNS value
        sess.run(DGNS_init)
        # train the diagnose network
        for step in range(DGNS_steps):
            #sampling
            samples,_ = balanced_sampling(L_train,DGNS_batch_size,pre_list = L_list)
            #reshape tensor
            x,y,_ = reshape_to_feed(X_train,L_train,samples, window_size,
                                    slice_map = slice_map,class_map = class_map)
            # Run optimization op (backprop)
            sess.run(DGNS_train, feed_dict={X: x, Y: y})
        # Calculate the prediction result
        for samples in chunks(X_train.keys()):
            x,y,key_list = reshape_to_feed(X_train,L_train,samples, window_size,
                                           slice_map = slice_map,class_map = class_map)
            # label the prediction with the most suspicous code
            SLCT_label = sess.run(DGNS_correct_pred,feed_dict={X: x,Y: y})
            # construct training label for SLCT
            i = 0
            for key in key_list:
                R_train[key] = int(SLCT_label[i])
                # update score_map
                update_score(key,float(SLCT_label[i]),alpha,meta_step)
                i += 1
        # train SLCT network,with updated prediction corectness result
        R_list = pre_classify(R_train,class_list = class_list)
        for step in range(SLCT_steps):
            #sampling the label
            samples,_ = balanced_sampling(R_train,SLCT_batch_size,pre_list = R_list)
            #reshape tensor
            x,y,_ = reshape_to_feed(X_train,R_train,samples, window_size,
                                    slice_map = slice_map,class_num = SLCT_classes)
            # Run optimization op (backprop)
            sess.run(SLCT_train, feed_dict={X: x,Y_S: y})
            
        # Apply trained network to update slice list
        change = 0.0
        for key in X_train.keys():
            # reshape to feed, x,y
            x = expand_slices(X_train[key],window_size)
            # label the prediction to check weather it is correct
            new_slice = sess.run(SLCT_argmax,feed_dict={X: x})
            if new_slice != slice_map[key]:
                # update the slice with certain shift rate
                slice_map[key] = new_slice
                change += 1.0
                
        change = change/len(X_train.keys())
        
        # Calculate batch loss and accuracy validate batch
        samples,_ = balanced_sampling(L_test,validate_batch, pre_list = V_list)
        # parse the most likely slice
        slice_map_V = {}
        for key in samples:
            # reshape to feed, x,y
            x = expand_slices(X_test[key],window_size)
            # label the prediction to check weather it is correct
            slice_map_V[key] = int(sess.run(UA_argmax,feed_dict={X: x}))
        # feed slice to diagnose network
        x,y,_ = reshape_to_feed(X_test,L_test,samples, window_size,
                          slice_map = slice_map_V, class_map = class_map)
        #update_meta_step
        meta_step += 1
        # run the loss and accuracy
        loss, acc = sess.run([DGNS_loss, DGNS_acc], feed_dict={X: x,Y: y})
        print("Meta Step " + str(meta_step) + ", Mean_Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc) + ", Change Rate= "+ \
              "{:.3f}".format(change))
        loss_list.append(loss)
        acc_list.append(acc)
        change_list.append(change)
        #Save the graph during training
        if meta_step % display_step == 0 or acc > 0.94:
            np.save('models/RRNN_v3/T1/loss_list_'+str(window_size)+'-'+str(meta_step)+'.npy',loss_list)
            np.save('models/RRNN_v3/T1/acc_list_'+str(window_size)+'-'+str(meta_step)+'.npy',acc_list)
            np.save('models/RRNN_v3//T1/change_list_'+str(window_size)+'-'+str(meta_step)+'.npy',change_list)
            saver = tf.train.Saver()
            saver.save(sess,'models/RRNN_v3/T1/RRNN_'+str(window_size),global_step = meta_step)
            np.save('models/RRNN_v3/T1/RRNN_'+str(window_size)+'-'+str(meta_step) +'_slice.npy', slice_map)
            np.save('models/RRNN_v3/T1/RRNN_'+str(window_size)+'-'+str(meta_step)+'_score.npy',score_map)
    print("First stage Learning Finished!")
    
    #% second stage training
    print("Start Second Stage training")
    R_squared_list=[]
    S_train = {}
    start_step = meta_step
    while (change > 0.01 and meta_step < start_step + meta_step_2):
        # initialize DGNS value
        sess.run(DGNS_init)
        # train the diagnose network
        for step in range(DGNS_steps):
            #sampling
            samples,_ = balanced_sampling(L_train,DGNS_batch_size,pre_list = L_list)
            #reshape tensor
            x,y,_ = reshape_to_feed(X_train,L_train,samples, window_size,
                                    slice_map = slice_map,class_map = class_map)
            # Run optimization op (backprop)
            sess.run(DGNS_train, feed_dict={X: x, Y: y})
            
        # Calculate the prediction result
        for samples in chunks(X_train.keys()):
            x,y,key_list = reshape_to_feed(X_train,L_train,samples, window_size,
                                           slice_map = slice_map,class_map = class_map)
            # label the prediction to check weather it is correct
            SLCT_label = sess.run(DGNS_correct_pred,feed_dict={X: x,Y: y})
            # construct training label for SLCT
            i = 0
            for key in key_list:
                # update score_map
                update_score(key,float(SLCT_label[i]),alpha,meta_step)
                i += 1
        #construct S_train
        for key in score_map.keys():
            S_train[key] = np.log10(score_map[key].astype(float)+0.00001)
        # initialize UA value
        sess.run(UA_init)
        # train universal approximator
        for step in range(UA_steps):
            # Train the universal approximator
            x,y = random_slice_pick(X_train,S_train,UA_batch_size)
            sess.run(UA_train,feed_dict = {X: x, SC: y})

        # Apply trained network to update slice list
        change = 0.0
        for key in X_train.keys():
            # reshape to feed, x,y
            x = expand_slices(X_train[key], window_size)
            # label the prediction to check weather it is correct
            new_slice = sess.run(UA_argmax,feed_dict={X: x})[0]
            if new_slice != slice_map[key]:
                # update the slice with certain shift rate
                slice_map[key] = int(new_slice)
                change += 1.0
                
        change = change/len(X_train.keys())
        
        # Calculate batch loss and accuracy validate batch
        samples,_ = balanced_sampling(L_test,validate_batch, pre_list = V_list)
        # parse the most likely slice
        slice_map_V = {}
        for key in samples:
            # reshape to feed, x,y
            x = expand_slices(X_test[key],window_size)
            # label the prediction to check weather it is correct
            slice_map_V[key] = int(sess.run(UA_argmax,feed_dict={X: x}))
        # feed slice to diagnose network
        x,y,_ = reshape_to_feed(X_test,L_test,samples, window_size,
                          slice_map = slice_map_V, class_map = class_map)
        # run the loss and accuracy
        loss, acc = sess.run([DGNS_loss, DGNS_acc], feed_dict={X: x,Y: y})
        # run R squared
        x,y = random_slice_pick(X_train,S_train,validate_batch)
        R_squared = sess.run(UA_R_squared,feed_dict = {X: x, SC: y})
        
        #update_meta_step
        meta_step += 1
        print("Step " + str(meta_step) + ", Loss= " + \
              "{:.4f}".format(loss) + ", Accuracy= " + \
              "{:.3f}".format(acc) + ", Change= "+ \
              "{:.3f}".format(change)+", R_squared = "+ \
              "{:.3f}".format(R_squared))
        loss_list.append(loss)
        acc_list.append(acc)
        change_list.append(change)
        R_squared_list.append(R_squared)
        #Save the graph during training
        if meta_step % display_step == 0 or acc > 0.94:
            np.save('models/RRNN_v3/T1/loss_list_'+str(window_size)+'-'+str(meta_step)+'.npy',loss_list)
            np.save('models/RRNN_v3/T1/acc_list_'+str(window_size)+'-'+str(meta_step)+'.npy',acc_list)
            np.save('models/RRNN_v3/T1/change_list_'+str(window_size)+'-'+str(meta_step)+'.npy',change_list)
            np.save('models/RRNN_v3/T1/R_squared_list_'+str(window_size)+'-'+str(meta_step)+'.npy',R_squared_list)
            saver = tf.train.Saver()
            saver.save(sess,'models/RRNN_v3/T1/RRNN_'+str(window_size),global_step = meta_step)
            np.save('models/RRNN_v3/T1/RRNN_'+str(window_size)+'-'+str(meta_step) +'_slice.npy', slice_map)
            np.save('models/RRNN_v3/T1/RRNN_'+str(window_size)+'-'+str(meta_step)+'_score.npy',score_map)
        
#% Final Training  
    print("Final training for diagnose network")
    for step in range(DGNS_steps_final):
        #sampling
        samples,_ = balanced_sampling(L_train,DGNS_batch_size,pre_list = L_list)
        #reshape tensor
        x,y,_ = reshape_to_feed(X_train,L_train,samples,window_size,
                                slice_map = slice_map,class_map = class_map)
        # Run optimization op (backprop)
        sess.run(DGNS_train, feed_dict={X: x, Y: y})
    # Calculate batch loss and accuracy validate batch
    samples,_ = balanced_sampling(L_train,validate_batch,pre_list = L_list)
    #reshape tensor
    x,y,_ = reshape_to_feed(X_train,L_train,samples, window_size,
                            slice_map = slice_map,class_map = class_map)
    # run the Final loss and accuracy
    loss, acc = sess.run([DGNS_loss, DGNS_acc], feed_dict={X: x,Y: y})
    print("Final Result: Mean_Loss= " + "{:.4f}".format(loss) + \
          ", Training Accuracy= " + \
          "{:.3f}".format(acc))
    
    print("Final training for selection network")
    #construct S_train
    for key in score_map.keys():
        S_train[key] = np.log10(score_map[key].astype(float)+0.00001)
    
    sess.run(UA_init)
    for step in range(UA_steps):
        # Train the universal approximator
        x,y= random_slice_pick(X_train,S_train,UA_batch_size)
        sess.run(UA_train,feed_dict = {X: x, SC: y})
        if (step+1) % 100 == 0:
            loss, R_squared = sess.run([UA_loss, UA_R_squared],feed_dict = {X: x, SC: y})
            print('Step '+ str(step+1) + \
                  ', Loss = {:.4f}'.format(loss) + \
                  ', R_squared = {:.4f}'.format(R_squared))
    #Now, save the graph
    saver = tf.train.Saver()
    saver.save(sess,'models/RRNN_v3/T1/final_'+str(window_size),global_step = meta_step)
    np.save('models/RRNN_v3/T1/final_slice_'+str(window_size)+'-'+str(meta_step), slice_map)
    np.save('models/RRNN_v3/T1/final_score_'+str(window_size)+'-'+str(meta_step), score_map)
#plot training result
#plt.figure()
#plt.plot(range(0,len(loss_list)),loss_list,'-')
#plt.title('Loss Function')
#plt.figure()
#plt.plot(range(0,len(acc_list)),acc_list,'-')
#plt.title('Training Accuracy')
#plt.figure()
#plt.plot(range(0,len(change_list)),change_list,'-')
#plt.title('Selection Shift Rate')