def image_load(path):
    image = cv.imread(path)
    image[:,:,[0,2]] = image[:,:,[2,0]]
    return image

def data_locations(directory, dataset_type = 'train'):
    """
    for internal usage
    """
#     directory = 'F:/EE5179_Course_Project/Denoising_Dataset_train_val/Denoising_Dataset_train_val'

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
                if dataset_type != 'test':
                    clean_file_locations.append(f"{directory}/{file}/{dataset_type}/GT_clean_image/{defect}/{file_name}")
                    
                degraded_file_locations.append(f"{directory}/{file}/{dataset_type}/Degraded_image/{defect}/{file_name}")
                
                name, extension = file_name.split('.')
                file_name1 = f'{name}_mask.{extension}'
                
                defect_mask_file_locations.append(f"{directory}/{file}/{dataset_type}/Defect_mask/{defect}/{file_name1}")
                
    if dataset_type == 'test':
        return degraded_file_locations, defect_mask_file_locations
    
    return clean_file_locations, degraded_file_locations, defect_mask_file_locations


def data_loading(directory, dataset_type = 'train'):
    """
    directory: str
        directory of the input data in the given format
        
    process_type: str ('train', 'val', 'test')
        for process_type 'test', clean files won't be available
    """
    if dataset_type == 'test':
        degraded_loc, defect_mask_loc = data_locations(directory = directory, dataset_type = dataset_type)
    else:
        clean_loc, degraded_loc, defect_mask_loc = data_locations(directory = directory, dataset_type = dataset_type)
        
    data = []
    for num in tqdm(range(len(degraded_loc)), desc = f'{dataset_type} Data'): 
            
        degraded_image = image_load(degraded_loc[num])
        defect_mask  = image_load(defect_mask_loc[num])
    
        if dataset_type != 'test':
            clean_image = image_load(clean_loc[num])
            data.append((clean_image, degraded_image, defect_mask))
        elif dataset_type == 'test':
            data.append((degraded_image, defect_mask))
            
    return data