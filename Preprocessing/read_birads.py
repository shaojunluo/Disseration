#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:57:31 2017

@author: shaojun
"""

import pandas as pd
#import numpy as np
from os.path import splitext
from os import listdir
import funclib as fl

folder_list = pd.read_csv('folder_list.csv')
BIRAD_data = pd.read_excel('CCNY MSK_16-328_BIRADS.xlsx',header = [0,1],index_col=0,sheetname='Sheet1')
#% filtered exist list
#load mri_file
mri_folder = '/media/shaojun/bottleneck_3/T1_bottleneck'
file_list = [splitext(f)[0] for f in listdir(mri_folder)]
file_list = fl.convertFileList(file_list)
file_info = folder_list.loc[folder_list['Folder'].isin(file_list),:]
file_info['key'] = file_info['ID'] + '_' + file_info['Date'].map(str)
# filter BIRADS data
label = pd.DataFrame(columns = BIRAD_data['MR 1'].columns)
for key in list(BIRAD_data.columns.levels[0]):
    subtable = BIRAD_data[key].reset_index()
    label = pd.concat([label,subtable],ignore_index = True)
label = label.loc[pd.notnull(label['BIRADS']),:]
label = label.loc[~label['BIRADS'].isin(['NR']),:]
# concat datetime table
label['key'] = label['index'] + '_' +label['completedDTTM'].apply(lambda s: s.strftime('%Y%m%d'))
#% join two tables
all_info_raw = file_info.merge(label, on = 'key')
#% clean final table
all_info = all_info_raw.drop(['index','Date'],axis = 1)
#% split left and right
for index, row in all_info.iterrows():
    # if has multiple BIRADS
    birads = str(row['BIRADS']).replace('; ',',').replace('.',',')
    if len(birads)>2:
        # check left or right
        if any([x in row['Type'].lower() for x in ['lt','left']]):
            try:
                all_info = all_info.set_value(index,'BIRADS', int(birads.split(',')[0]))
            except:
                print('invalid BIRADS: '+birads)
                all_info = all_info.drop(index)
                continue
        elif any([x in row['Type'].lower() for x in ['rt','right']]):
            all_info = all_info.set_value(index,'BIRADS', int(birads.split(',')[1]))
        else:
            print('No side determined in: '+ row['Type'] + ' BIRADS: ' + str(row['BIRADS']))
            # drop bad point
            #all_info = all_info.drop(index)
            continue
    else:
        all_info = all_info.set_value(index,'BIRADS',int(row['BIRADS']))
    # drop unformatted values
    #if all_info.loc[index,'BIRADS']>6:
    #    all_info = all_info.drop(index)

all_info.to_csv('T1_label.csv',index=False)
