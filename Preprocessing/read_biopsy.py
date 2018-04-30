#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:35:37 2017

@author: shaojun
"""

#from shutil import copytree
import pandas as pd

#%% Process the unstructured file

PATH_data = pd.read_excel('CCNY MRI PTs_Batch 7_old.xlsx',header = [0,1],index_col=0)

data = pd.DataFrame(columns = PATH_data['MR 1'].columns)
for key in list(PATH_data.columns.levels[0]):
    subtable = PATH_data[key].reset_index()
    data = pd.concat([data,subtable],ignore_index = True)
data = data.rename(columns = {'index':'DE-ID'})
data = data.loc[pd.notnull(data['Pathology']),:]

data.to_excel('CCNY MRI PTs_Batch 7.xlsx',index = False)

#%% Combine all structured files 

#read T1 file
label = pd.read_csv('T1_label.csv')

# combine biopsy files
n_batches  = 7
biopsy = []
for i in range(n_batches):
    biopsy.append(pd.read_excel('CCNY MRI PTs_Batch ' + str(i+1) +'.xlsx'))
biopsy = pd.concat(biopsy)
biopsy = biopsy.loc[biopsy['Completion Date'].notnull()]
#src = '/media/shaojun/My Passport'
#dst = '/media/shaojun/DICOM_selected/DICOM'
biopsy['key'] = biopsy['DE-ID'] + '_' +biopsy['Completion Date'].apply(lambda s: s.strftime('%Y%m%d'))
full = biopsy.merge(label[['key','Folder','Type']], on = ['key'])

#delete mismatch
for index,row in full.iterrows():
    if row['Side'] == 'Left':
        if any([x in row['Type'].lower() for x in ['rt','right']]):
            # drop conflict data
            full = full.drop(index)
    elif row['Side'] == 'Right':
        if any([x in row['Type'].lower() for x in ['lt','left']]):
            # drop conflict data
            full = full.drop(index)
    birads = str(row['BIRADS']).replace('; ',',').replace('.',',')
    # if has multiple BIRADS
    if len(birads)>2:
        # check left or right
        if any([x in row['Type'].lower() for x in ['lt','left']]):
            try:
                # assign left BIRADS
                full = full.set_value(index,'BIRADS', int(birads.split(',')[0]))
            except:
                # assign right BIRADS
                print('invalid BIRADS: '+birads)
                full = full.drop(index)
                continue
        elif any([x in row['Type'].lower() for x in ['rt','right']]):
            full = full.set_value(index,'BIRADS', int(birads.split(',')[1]))
        else:
            print('No side determined in: '+ row['Type'] + ' BIRADS: ' + str(row['BIRADS']))
            # drop bad point
            full = full.drop(index)
            continue
# drop duplicate        
full = full.drop_duplicates()
#% drop null data
full = full.dropna(subset = ['Pathology'])
full.to_csv('path_03_21_t1.csv')
