#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:05:59 2017

@author: shaojun
"""


import pandas as pd


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

list_file = 'folder_list_3.csv'
folder_table = pd.read_csv(list_file)
# look at the list
folder_list = filter_folder_list(folder_table)
x = list(folder_list['Type'].unique())

x = [c for c in x if ( ('ax' not in c.lower()) & 
                      (('po' in c.lower()) | ('pro' in c.lower())) &
                      ('t1' in c.lower()) &
                      ('map' not in c.lower()))]

type_list = pd.DataFrame(x,columns = ['Type'])
type_list.to_csv('T1_post_list_3.csv')
