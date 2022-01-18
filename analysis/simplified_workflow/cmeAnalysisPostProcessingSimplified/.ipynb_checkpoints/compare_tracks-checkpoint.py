# Cyna Shirazinejad 1/17/22
import os
from skimage import feature, exposure, io
from skimage.color import rgb2gray
from skimage.filters import median
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
def calculate_fraction_area_occupied_by_cells(path_tracks,
                                              identifier_string,
                                              channel,
                                              imagesize=[512,512]):
    
    all_cell_mask_cell_pixel_fraction = []
    
    all_track_paths = os.listdir(path_tracks)
    all_track_paths = [exp for exp in all_track_paths if identifier_string in exp]
    all_track_paths.sort()
    print('\nfolders to mine:')
    for exp_name in all_track_paths:    
        print(exp_name)
    print('\n')
    
    for i, track_path in enumerate(all_track_paths):
        
        current_channel_paths = path_tracks +'/'+ track_path + '/Ch' + str(channel)
        
        possible_files = os.listdir(current_channel_paths)
#         print(possible_files)
        for file in possible_files:
            
            if '.tif' in file:
                
                filename = file
                
        raw_image = io.imread(path_tracks + 
                              '/'+
                              track_path + 
                              '/Ch' + 
                              str(channel) + 
                              '/' + 
                              filename)[0]
        
        grayscale_image = rgb2gray(raw_image)
        grayscale_image=median(grayscale_image,disk(5))
        percentile_threshold = np.percentile(grayscale_image.flatten(),10)
        edges = feature.canny(grayscale_image>percentile_threshold, sigma=1)

        plt.imshow(grayscale_image>percentile_threshold,alpha=0.08)
        plt.imshow(exposure.adjust_gamma(raw_image, 0.00001),alpha=0.9)
        plt.imshow(edges,alpha=0.5)

        plt.show()

        all_cell_mask_cell_pixel_fraction.append(
            (grayscale_image>percentile_threshold).
            sum()/(imagesize[0]*imagesize[1]))
        
    return all_cell_mask_cell_pixel_fraction