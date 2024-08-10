import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import shutil

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-recon_npy_path", "--recon_npy_path", help="The object reconstructured path", default=f"./recon_object_array")
parser.add_argument("-ref_npy_path", "--ref_npy_path", help="The raw cropped object path", default=f"./raw_object_array")
parser.add_argument("-residual_output_path", "--residual_output_path", help="The output residual npy path",  default=f"./residual_array")
args = parser.parse_args()


def is_image_file(filename): # Compares 'filename' extension to common image file types.
    return any(filename.endswith(extension) for extension in ['.npy'])
def load_image_path(imgDir):

    all_training_files=os.walk(imgDir)
    train_files=[]
    train_imageNames=[]
    train_nSamples=0
    for path,direction,filelist in all_training_files:
        files = [file for file in filelist if os.path.isfile(os.path.join(path, file))]
        imageNames = [file.split('.')[0] for file in files if is_image_file(file)]
        files = [os.path.join(path, file) for file in files if is_image_file(file)]
        train_files.append(files)
        train_imageNames.append(imageNames)
        train_nSamples=train_nSamples+len(files)
    train_files=sum(train_files,[])
    train_imageNames=sum(train_imageNames,[])
    #print(train_imageNames[0])
    #print(train_files[0])
    
    return train_files, train_imageNames

def get_all_folder_paths(directory):
    file_paths = []
    dir_list = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            dir_list.append(folder_path)
    dir_list.sort()
    return dir_list

reconstructured_path = args.recon_npy_path
reference_image_path = args.ref_npy_path

output_file_path = args.residual_output_path

recon_image_path = get_all_folder_paths(reconstructured_path)
ref_image_path = get_all_folder_paths(reference_image_path)

#recon_image_path = os.walk(reconstructured_path)
#ref_image_path = os.walk(reference_image_path)  
iteration = 0
for path in ref_image_path:
#for path,direction,filelist in ref_image_path:
    iteration = iteration + 1
    #print("the iteration is: ", iteration)
    if (iteration != 1):
        temp_list = path.split('/')
        object_theme = temp_list[-1]
        #print(path)
        ref_image_path_new = path
        
        recon_image_path_new = reconstructured_path + '/' + object_theme + '/'
        #print(recon_image_path_new)
        recon_image_files, recon_image_names = load_image_path(recon_image_path_new)
        
        ref_image_files, ref_image_names  = load_image_path(ref_image_path_new)
        #print(ref_image_files[0:5])
        
        recon_image_num = len(recon_image_files)
        
        output_path = output_file_path + '/' + object_theme + '/'
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        for i in tqdm(range (recon_image_num)):
            
            output_path_npy = output_path + f'cropped_object_{i}.npy'
            
            recon_image = np.load(recon_image_files[i])
            
            ref_image = np.load(ref_image_files[i])
            
            recon_image_int8  = np.int8(recon_image)
            ref_image_int8 = np.int8(ref_image)
            residual_image = recon_image_int8 - ref_image_int8
            np.save(output_path_npy, residual_image)