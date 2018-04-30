# Disseration
Codes for dissertation Chapter 4

Folder and usages:

Folder -- get_bottlenecks:

Feature Extraction Agent. Extract features of dicom files with inception_v3 CNN. bottlenecks.py automatically download and run the models on MRI images stored as dicom files in a folder (1 scan per folder). The output is the feature map matrix n*2048 where n is the number of dicom files and 2048 is the number of feature extracted
