#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:25:40 2017

@author: shaojun
"""
import os
import dicom
import pandas as pd
from tensorflow.python.platform import gfile

def create_image_lists(image_dir):
    """Builds a list of training images from the file system.
    
    	Analyzes the sub folders in the image directory, splits them into stable
    	training, testing, and validation sets, and returns a data structure
    	describing the lists of images for each label and their paths.
    
    	Args:
    		image_dir: String path to a folder containing subfolders of images.
    		testing_percentage: Integer percentage of the images to reserve for tests.
    		validation_percentage: Integer percentage of images reserved for validation.
    
    	Returns:
    		A dictionary containing an entry for each label subfolder, with images split
    		into training, testing, and validation sets within each label.
    """
    #initiate folders
    folder_table = pd.DataFrame(columns = ['Folder','ID','Code','Type','Date','Size','Slices'])
    # make sure it exist
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    # get the list of subjects
    sub_list = [image_dir+'/'+sub_dir for sub_dir in next(os.walk(image_dir))[1]]
    n_subs = len(sub_list)
    print('Number of subjects:{0:d}'.format(n_subs))
    #iterate over subjects
    count_folder = 0
    count_sub = 0.0
    error_list = []
    for subs in sub_list:
        count_sub +=1
        #look into each subject
        for x in gfile.Walk(subs):
            #if we have file inside the folder
            if len(x[2])>0:
                # check if it contains dicom files
                n_z = len(x[2])
                if x[2][0].endswith('.dcm'):
                    try:
                        dFile = dicom.read_file(x[0]+'/'+ x[2][0],force = True)
                        folder_temp = {}
                        folder_temp['Folder'] = x[0].replace(image_dir,'')
                        folder_temp['ID'] = dFile.PatientID
                        folder_temp['Code'] = dFile.SeriesNumber
                        folder_temp['Type'] = dFile.SeriesDescription
                        folder_temp['Date'] = dFile.StudyDate
                        folder_temp['Size'] = dFile.Rows
                        folder_temp['Slices'] = n_z
                        # attach folder
                        folder_table =folder_table.append(folder_temp,ignore_index=True)
                        count_folder += 1
                        #dynamically save the result
                        if count_folder%100 == 0:
                            folder_table.to_csv('folder_list_1.csv',index=False)
                        print('{0:d} scans detected (~{1:.2f}%)\r'.format(count_folder,100*count_sub/n_subs)),
                    except:
                        print('error\r'),
                        error_list.append(x[0]+'/'+x[2][0]) 
                        continue
    print('Folder Search Completed')
    return folder_table,error_list
#end code 
    
if __name__ == '__main__':
    #dicom_1
    image_dir = '/media/shaojun/9bd23e2c-3865-3928-80cd-95ebac2df1d4/DICOM'
    #dicom_2
    #image_dir = '/media/shaojun/850ef5d4-d989-3ce0-b984-ad99886afb4f/DICOM2'
    #dicom_3
    #image_dir = '/media/shaojun/My Passport'
    # dicom test
    #image_dir = '/home/shaojun/projects/Breast_MRI'
    folder_table,error_list = create_image_lists(image_dir)
    print('Save list')
    folder_table.to_csv('folder_list_1.csv',index=False)
    pd.DataFrame(error_list,columns = ['Path']).to_csv('error_list_1.csv',index=False)
    print('Complete')
