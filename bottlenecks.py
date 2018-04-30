#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:05:21 2017

@author: shaojun
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from datetime import datetime
#import hashlib
import os.path
import sys
import tarfile
from PIL import Image
import dicom
import pandas as pd

import numpy as np
from six.moves import urllib
from io import BytesIO
import tensorflow as tf

#from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
#from tensorflow.python.util import compat
# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def convert_dcm_to_jpeg(image_path):
    dFile=dicom.read_file(image_path)
    im = Image.fromarray(dFile.pixel_array)
    f = BytesIO()
    # save to jpeg file
    im.convert('RGB').save(f, 'jpeg')
    #return the value string
    return f.getvalue()

def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
    Args:
        dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def maybe_download_and_extract(model_dir):
    """Download and extract model tar file.
    
    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_inception_graph(model_dir):
    with tf.Session() as sess:
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                    tf.import_graph_def(graph_def, name='', return_elements=[
                            BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                            RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor,bottleneck_tensor):										
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def save_bottleneck_file(bottleneck_values,bottleneck_dir,folder):
    #simplified the bottleneck path
    file_name = '_'.join([c for c in folder.split('/') if c.isdigit()])
    bottleneck_path = bottleneck_dir+'/'+file_name
    np.save(bottleneck_path,bottleneck_values)
    
def create_bottleneck(sess, folder, file_name, jpeg_data_tensor, bottleneck_tensor):
    # get full path
    image_path = folder + '/' + file_name
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    #convert to jpg image data return as binary file
    image_data = convert_dcm_to_jpeg(image_path)
    bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    # return the numpy array of bottlenecks
    return bottleneck_values
    #create bottleneck paths

def cache_bottlenecks(sess, folder_list, image_dir, bottleneck_dir,
											jpeg_data_tensor, bottleneck_tensor):
    error_list = []
    n_bottlenecks = 0
    n_scans = len(folder_list)
    print('number of scans:'+str(n_scans))
    for folder in folder_list:
        # list for stack
        bottleneck_values = []
        folder_full = image_dir+ '/' + folder
        for file_name in os.listdir(folder_full):
            # creat bottleneck file
            try:
                bottleneck_values.append(
                        create_bottleneck(sess,folder_full,file_name,
                                          jpeg_data_tensor, bottleneck_tensor))
            except:
                print('\n error in '+folder_full+'/'+file_name)
                error_list.append(folder_full+'/'+file_name)
        #stack bottlenecks
        bottleneck_values = np.asarray(bottleneck_values)
        #save bottlenecks
        save_bottleneck_file(bottleneck_values,bottleneck_dir,folder)
        n_bottlenecks += 1
        print('{0:d} scans processed ({1:.2f}%) \r'.format(n_bottlenecks,100.0*n_bottlenecks/n_scans)),
    pd.DataFrame(error_list,columns =['Path']).to_csv('b_error_list.csv',index=False)
            
def filter_folder_list(folder_table, 
                       Type_list = None, ID_list = None, Date_list = None,
                       Size_list = None, Slice_list = None):
    fil = [True]*len(folder_table)
    if Type_list is not None:
        fil = fil & folder_table['Type'].isin(Type_list)
    if ID_list is not None:
        fil = fil & folder_table['ID'].isin(ID_list)
    if Date_list is not None:
        fil = fil & folder_table['Date'].isin(Date_list)
    if Size_list is not None:
        fil = fil & folder_table['Size'].isin(Size_list)
    if Slice_list is not None:
        fil = fil & folder_table['Slice'].isin(Slice_list)
        
    return folder_table.loc[fil][:]

#initiate input
model_dir = '/tmp/imagenet'
#dicom 1
image_dir = '/media/shaojun/My Passport/DICOM2'
# dicom_2
#image_dir = '/media/shaojun/850ef5d4-d989-3ce0-b984-ad99886afb4f/DICOM2'
# image_test
#image_dir = '/home/shaojun/projects/Breast_MRI'

list_file = 'folder_list_2.csv'
#list_file = 'folder_list_test.csv'
bottleneck_dir = '/media/shaojun/bottleneck_3/T1_bottleneck'

ID_list = None
Date_list = None
folder_list = pd.read_csv(list_file)
Type_list = list(pd.read_csv('T1_post_list_2.csv')['Type'])

ensure_dir_exists(bottleneck_dir)
# Set up the pre-trained graph.
maybe_download_and_extract(model_dir)
graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph(model_dir))

# filter the list
pull_list = filter_folder_list(folder_list,
                               Type_list = Type_list, 
                               ID_list = ID_list,
                               Date_list = Date_list)
# look at the list
folder_list = pull_list['Folder']

sess = tf.Session()
#
cache_bottlenecks(sess,folder_list, image_dir, bottleneck_dir,
                  jpeg_data_tensor, bottleneck_tensor)

print('Caching Complete')
