#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:35:51 2017

@author: shaojun
"""
import numpy as np
from numpy import matlib as npm
from numpy import random as rnd
import pandas as pd
import networkx as nx
import community as com
import scipy as sp
from scipy import signal as sgl

# fast Clustering use Louvain Modularity to fast clustering the graph
# return the diction 
def fastClustering(G,data_cluster,data_plot = None, n_order = None):
    #parsing input
    if data_plot is None:
        data_plot = data_cluster
    # positive weight for clustering
    Jij = zip(*G.edges(data = data_cluster))
    Jij[2] = np.abs(Jij[2])
    # update atrritbute to calculate modularity
    for (u,v) in G.edges(data=False):
        G[u][v]['abs_weight']=np.abs(G[u][v][data_cluster])
    # calculate modularity
    coms = com.best_partition(G,weight = 'abs_weight')
    # parsing if clustered
    if n_order is None:
        # clustered by modularity
        n_old = np.array(coms.keys())
        n_com = np.array(coms.values())
        # get new index use indirect sorting
        n_order = n_old[np.argsort(n_com)]
    # generate sorted adjMat
    AdjMat = nx.adjacency_matrix(G, nodelist = n_order, weight=data_plot)

    return coms, n_order, AdjMat
# correlationmmatrix with output p value

def norm(v):
    fil = ~np.isnan(v)
    return np.float64((v - np.min(v[fil]))/(np.max(v[fil]) - np.min(v[fil])))

def smooth(image,x_window,y_window):
    img = np.copy(image)
    img[np.isnan(img)] = 0
    window = np.float64(np.ones((x_window,y_window)))/(x_window*y_window)
    return sgl.convolve2d(img,window,mode = 'same')

def convertFileList(file_list):
    # add prefix
    out_list = list(map(lambda s: '/'+s,file_list))
    # add subfix
    out_list = list(map(lambda s: s+'/DICOM',out_list))
    # replace center
    out_list = list(map(lambda s: s.replace('_', '/SCANS/'),out_list))
    return out_list

def convertFolderList(folder_list):
    # add prefix
    out_list = list(map(lambda s: s.replace('/',''),folder_list))
    # add subfix
    out_list = list(map(lambda s: s.replace('DICOM',''),out_list))
    # replace center
    out_list = list(map(lambda s: s.replace( 'SCANS','_'),out_list))
    return out_list
