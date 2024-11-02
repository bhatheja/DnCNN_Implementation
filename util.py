import os
from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        for process_type 'Test', clean files won't be available

    Returns
    -------
    data: tuple
        tuple having lists of clean image, degraded image, and corresponding mask image
    """
    if dataset_type == 'Test':
        degraded_loc, defect_mask_loc = data_locations(directory = directory, dataset_type = dataset_type)
    else:
        clean_loc, degraded_loc, defect_mask_loc = data_locations(directory = directory, dataset_type = dataset_type)
        
    data = []
    for num in tqdm(range(len(degraded_loc)), desc = f'{dataset_type} Data'): 
            
        degraded_image = image_load(degraded_loc[num])
        defect_mask  = image_load(defect_mask_loc[num])
    
        if dataset_type != 'Test':
            clean_image = image_load(clean_loc[num])
            data.append((clean_image, degraded_image, defect_mask))
        elif dataset_type == 'Test':
            data.append((degraded_image, defect_mask))
            
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

def patch_saving(patch_size_info = 128, data_type='train', folder_location = 'should_be_given'):
    """
    Input
    -----
    patch_size_info: int
        the patch size in which the images to be stored
    data_type: str
        It can take 3 values 'train', 'val', 'test' but it does not make sense to save the patch for validation and test data unless we have efficiency issue

    Returns- None
    -------------
    """
    print(f'Saving {data_type} images of patch size of {patch_size_info}')
    file_size_info = []
    for tp in range(3):
        count = 0
        # Locations to save the file
        if tp==0:
            location = f'{folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}/clean'
        elif tp==1:
            location = f'{folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}/degraded'
        elif tp==2:
            location = f'{folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}/mask'
            
        os.makedirs(location, exist_ok = True)
        
        if data_type=='train':
            for i in tqdm(range(len(train_data)), desc = f'Type:{tp}'):
                count+=1
                patches = create_vertical_and_horizontal_patches(train_data[i][tp], patch_size = patch_size_info)

                # Registering the image size for stitching 
                if tp==0:
                    height, width, _ = np.array(train_data[i][tp]).shape

                    file_size_info.append({'image_Num': count, 'height': height, 'width': width})
                for patch in range(len(patches)):
                    array = (patches[patch]*255).astype(np.uint8)

                    image = Image.fromarray(array)
                    image.save(f'{location}/{count}_{patch+1}.png')

        elif data_type=='val':
            for i in tqdm(range(len(val_data)), desc = f'Type:{tp}'):
                count+=1
                patches = create_vertical_and_horizontal_patches(val_data[i][tp], patch_size_info)

                # Registering the image size for stitching 
                if tp==0:
                    height, width, _ = np.array(val_data[i][tp]).shape

                    file_size_info.append({'image_Num': count, 'height': height, 'width': width})
                for patch in range(len(patches)):
                    array = (patches[patch]*255).astype(np.uint8)

                    image = Image.fromarray(array)
                    image.save(f'{location}/{count}_{patch+1}.png')

    df = pd.DataFrame(file_size_info)
    df.to_csv(f'{folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}/image_size_info_{patch_size_info}x{patch_size_info}.csv',
              index=False)
    print(f'Images Saved to {folder_location}/{patch_size_info}x{patch_size_info}_patches/{data_type}')


