import os
from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

# For Loading the image using open_cv
def image_load(path):
    image = cv.imread(path)
    image[:,:,[0,2]] = image[:,:,[2,0]]
    return image

# For loading the image using matplotlib for reading the mask
def image_load_plt(path):
    image = plt.imread(path)
    return image


# For getting the location of files for particular dataset setup
def data_locations(directory, dataset_type = 'Train'):
    """
    for internal usage
    """
    
    clean_file_locations = []; degraded_file_locations = []; defect_mask_file_locations = []

    with os.scandir(directory) as objects:
        files = [entry.name for entry in objects]
    
    for file in files:
        file_dir = f'{directory}/{file}/{dataset_type}/Degraded_image'
        with os.scandir(file_dir) as types:
            defect_type = [entry.name for entry in types]

        for defect in defect_type:
            with os.scandir(f'{file_dir}/{defect}') as file_name:
                file_names = [entry.name for entry in file_name]

            for file_name in file_names:
                if dataset_type != 'Test':
                    clean_file_locations.append(f"{directory}/{file}/{dataset_type}/GT_clean_image/{defect}/{file_name}")
                    
                degraded_file_locations.append(f"{directory}/{file}/{dataset_type}/Degraded_image/{defect}/{file_name}")
                
                name, extension = file_name.split('.')
                file_name1 = f'{name}_mask.{extension}'
                
                defect_mask_file_locations.append(f"{directory}/{file}/{dataset_type}/Defect_mask/{defect}/{file_name1}")
                
    if dataset_type == 'Test':
        return degraded_file_locations, defect_mask_file_locations
    
    return clean_file_locations, degraded_file_locations, defect_mask_file_locations


def data_loading(directory, dataset_type = 'Train'):
    """
    Input
    -----
    directory: str
        directory of the input data in the given format
        
    process_type: str ('Train', 'Val', 'Test')

    Returns
    -------
    data: tuple
        tuple having lists of clean image, degraded image, and corresponding mask image
    """

    clean_loc, degraded_loc, defect_mask_loc = data_locations(directory = directory, dataset_type = dataset_type)
        
    data = []
    for num in tqdm(range(len(degraded_loc)), desc = f'{dataset_type} Data'): 
            
        degraded_image = image_load(degraded_loc[num])
        defect_mask  = image_load(defect_mask_loc[num])
    
        clean_image = image_load(clean_loc[num])
        data.append((clean_image, degraded_image, defect_mask))
            
    return data


# Creating patches to make training easier (Making images of equal sizes
def create_vertical_and_horizontal_patches(image, patch_size=128):
    """
    Input
    -----
    image: Numpy array
        the image should have a dimension format (height, width, channels) where any height or width can not be more than 1024
        
    patch_size: int
        the patch size in which the image should be splitted

    Returns
    -------
    patches: list
        It returns the list of all patches for the input image
    """
    h, w, c = image.shape
    remaining_row = 1024-h
    remaining_col = 1024-w

    if remaining_row !=0:
        add_to_row = np.zeros((remaining_row, w, 3))
        image = np.concatenate((image, add_to_row), axis = 0)

    if remaining_col!=0:
        add_to_col = np.zeros((1024, remaining_col, 3))
        image = np.concatenate((image, add_to_col), axis = 1)

    image = image.astype(np.float32)/255.0
    
    patches = []
    for i in range(0, 1024, patch_size):
        for j in range(0, 1024, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches


# For saving patch files for training which is only relevant for training and not testing 

def patch_saving(data, patch_size_info = 128, data_type='train', folder_location = 'should_be_given'):
    """
    Input
    -----
    data: tuple
        it is having tuples of type (clean image, degraded image, defect mask image) for training purpose
    patch_size_info: int
        the patch size in which the images to be stored
    data_type: str
        It can take 3 values 'train', 'val', 'test' but it does not make sense to save the patch for validation and test data unless we have efficiency issue

    Returns- None
    -------------
    """
    print(f'Saving {data_type} images of patch size of {patch_size_info}')
    file_size_info = []
    for tp in range(2):
        count = 0
        # Locations to save the file
        if tp==0:
            location = f'{folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}/clean'
        elif tp==1:
            location = f'{folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}/degraded'
            
        os.makedirs(location, exist_ok = True)
        
        for i in tqdm(range(len(data)), desc = f'Type:{tp}'):
            count+=1
            patches = create_vertical_and_horizontal_patches(data[i][tp], patch_size = patch_size_info)

            # Registering the image size for stitching 
            if tp==0:
                height, width, _ = np.array(data[i][tp]).shape

                file_size_info.append({'image_Num': count, 'height': height, 'width': width})
            for patch in range(len(patches)):
                array = (patches[patch]*255).astype(np.uint8)

                image = Image.fromarray(array)
                image.save(f'{location}/{count}_{patch+1}.png')

    df = pd.DataFrame(file_size_info)
    df.to_csv(f'{folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}/image_size_info_{patch_size_info}x{patch_size_info}.csv',
              index=False)
    print(f'Images Saved to {folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}')

# Loading images (patches) from dataset

def patches_load(directory, patch_size = 128, total_images = 932):
    if patch_size == 128:
        total_patches = 64
    elif patch_size == 64:
        total_patches = 256

    patches=[]
    for image_no in tqdm(range(0,total_images)):
        for patch_idx in range(0,total_patches):
            clean = image_load(f'{directory}/clean/{image_no+1}_{patch_idx+1}.png')
            deg = image_load(f'{directory}/degraded/{image_no+1}_{patch_idx+1}.png')
            patches.append((clean, deg))
    return patches

# For image stitching back from patches
def retrieve_image(patches, patch_size=128, img_size_height=1024, img_size_width=1024):
    #Creating an empty image to hold the stitched result
    stitched_image = np.zeros((1024, 1024, 3), dtype=patches[0].dtype)
    retrieved_image = np.zeros((img_size_height, img_size_width, 3), dtype=patches[0].dtype)
    
    #Placing each patch in the correct location
    idx = 0
    for i in range(0, 1024, patch_size):
        for j in range(0, 1024, patch_size):
            stitched_image[i:i + patch_size, j:j + patch_size] = patches[idx]
            idx += 1
    retrieved_image = stitched_image[0:img_size_height, 0:img_size_width]
    return retrieved_image

# This is to get psnr value for training purpose and readjusting the learning rate
# psnr calculation
def cal_psnr_numpy(pred, clean, max_value=255.0):
    mse = np.mean((clean - pred) ** 2)
    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr

def psnr_values(dataloader, model, device, epoch):
    psnr = []
    for ite, (clean_image, noisy_image, _) in enumerate(dataloader):
        _, h, w, c = noisy_image.shape
        clean_image = clean_image.squeeze(0).detach().cpu().numpy()
        noisy_image = noisy_image.squeeze(0).detach().cpu().numpy()
        if epoch!=0:
            noisy_image = np.array(create_vertical_and_horizontal_patches(noisy_image))
            noisy_image = torch.from_numpy(noisy_image).permute(0,3,1,2).float().to(device)
            noisy_image = torch.nn.functional.pad(noisy_image, (1, 1, 1, 1, 0, 0, 0, 0), value=0)
            output = model(noisy_image)
            output = output[:, :, 1:-1, 1:-1]
            output = output.permute(0,2,3,1).detach().cpu().numpy()
            output = retrieve_image(output, patch_size=128, img_size_height = h, img_size_width = w)
            psnr.append(cal_psnr_numpy(output, clean_image))
        else:
            psnr.append(cal_psnr_numpy(noisy_image, clean_image))
    return psnr
        
    
