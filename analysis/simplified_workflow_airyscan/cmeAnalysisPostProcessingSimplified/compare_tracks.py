# Cyna Shirazinejad 1/17/22
import os
from skimage import feature, exposure, io
from skimage.color import rgb2gray
from skimage.filters import median
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import import_tracks
import generate_index_dictionary

def calculate_fraction_area_occupied_by_cells(path_tracks,
                                              identifier_strings,
                                              channel,
                                              imagesize=[512,512]):
    
    all_cell_mask_cell_pixel_fraction = []
    
    temp_paths = os.listdir(path_tracks)
    all_track_paths = []
    for exp in temp_paths:
        num_matches = 0
        for identifier in identifier_strings:
            if identifier in exp:
                num_matches+=1
        if num_matches==len(identifier_strings):
            all_track_paths.append(exp)
#     all_track_paths = [exp for exp in all_track_paths if identifier_string in exp]
    all_track_paths.sort()
    exp_num_index = [int(exp.split('_')[0]) for exp in all_track_paths]
    all_track_paths = [all_track_paths[idx] for idx in np.argsort(exp_num_index)]
       
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


def compare_components_between_conditions(analysis_metadata,
                                          conditions,
                                          fraction_areas,
                                          normalization_factor):

    number_of_clusters = analysis_metadata.item().get('number_of_clusters')
    
    exp_number_label = []
    cluster_label = []
    compiled_labels = []
    fraction_cluster_label = []
    rates = []
    
    for condition_number, current_condition in enumerate(conditions):
        
        for exp_index, exp_number in enumerate(set(current_condition['experiment_number'].values)):
            
            exp_number_label += [exp_number for _ in range(number_of_clusters)]
            cluster_label += list(range(number_of_clusters))
            compiled_labels += [condition_number for _ in range(number_of_clusters)]
            
            indices_experiment = current_condition[current_condition['experiment_number']==exp_number].index.values
            
            gmm_predictions_exp = current_condition['gmm_predictions'][indices_experiment].values
            
            for comp_number in range(number_of_clusters):
                
                num_in_comp = len(np.where(gmm_predictions_exp==comp_number)[0])
                
                fraction = num_in_comp/len(gmm_predictions_exp)
                print(fraction)
                fraction_cluster_label.append(fraction)
                rates.append(num_in_comp/fraction_areas[condition_number][exp_index]/normalization_factor)
                
    exp_number_label = np.array(exp_number_label).reshape(len(exp_number_label), -1).astype(np.float)
    cluster_label = np.array(cluster_label).reshape(len(exp_number_label), -1).astype(np.float)
    fraction_cluster_label = np.array(fraction_cluster_label).reshape(len(exp_number_label), -1).astype(np.float)
    compiled_labels = np.array(compiled_labels).reshape(len(exp_number_label), -1)
    rates = np.array(rates).reshape(len(exp_number_label), -1)
    
    data_df = np.hstack((exp_number_label, 
                         cluster_label, 
                         fraction_cluster_label, 
                         rates,
                         compiled_labels))
    df_fraction_comps_between_experiments = pd.DataFrame(data=data_df, columns=['experiment_number', 
                                                                                'component_number', 
                                                                                'fraction', 
                                                                                'rates',
                                                                                'condition'])        
    
    return df_fraction_comps_between_experiments
    
    
def compare_frequencies_all_track_categories(list_tracks,
                                             list_identifier_strings):
    
                
#     f = plt.figure(dpi=200, figsize=(10,10))
#     ax = f.add_subplot(1, 1, 1)
        
    index_dictionary = generate_index_dictionary.return_index_dictionary() # get indices of features in ProcessedTracks

    category_frequencies = []
    num_events_category = []
    categories = []
    track_set_identity = []
    
    for track_set_num, path_tracks in enumerate(list_tracks):
        
        temp_paths = os.listdir(path_tracks)
#         print(temp_paths)
        all_track_paths = []
        for exp in temp_paths:
            num_matches = 0
#             print(list_identifier_strings[track_set_num])
            for identifier in list_identifier_strings[track_set_num]:
                if identifier in exp:
                    num_matches+=1
            if num_matches==len(list_identifier_strings[track_set_num]):
                all_track_paths.append(exp)

        exp_num_index = [int(exp.split('_')[0]) for exp in all_track_paths]
        all_track_paths = [all_track_paths[idx] for idx in np.argsort(exp_num_index)]
                
        print('\nfolders to mine for track set {}:'.format(track_set_num+1))
        for exp_name in all_track_paths:    
            print(exp_name)
        print('\n')
        
        

        print('loading tracks in set..')
        
        all_tracks = []
        
        for track_num, tracks in enumerate(all_track_paths):
            print('tracks in set {} of {}'.format(track_num+1, len(all_track_paths)))
            current_tracks = import_tracks.load_tracks(path_tracks+'/'+tracks + '/' + '/Ch1/Tracking/ProcessedTracks.mat')
        
            all_tracks.append(current_tracks)
            
        # gather the event category (1-8) frequencies

        
#         x_std = [[] for i in range(len(all_tracks))]
        
        for track_set in all_tracks:

            num_tracks_exp = len(track_set)
            
            for category in range(1,9): # track categories available

                tracks_category_i_exp = import_tracks.remove_tracks_by_criteria(track_set, track_category=[category])
                
                num = len(tracks_category_i_exp)
                num_events_category.append(num)
                category_frequencies.append(num/num_tracks_exp)
                categories.append(category)
                track_set_identity.append(track_set_num)
                
        
        print('\n\n\n')
    num_events_category = np.array(num_events_category).reshape(len(num_events_category), -1).astype('float')
    category_frequencies = np.array(category_frequencies).reshape(len(category_frequencies), -1).astype('float')
    categories = np.array(categories).reshape(len(categories), -1).astype('float')
    track_set_identity = np.array(track_set_identity).reshape(len(track_set_identity), -1).astype('float')
    
    data_df = np.hstack((num_events_category,
                         category_frequencies, 
                         categories,
                         track_set_identity))
    
    df = pd.DataFrame(data=data_df, columns=['counts',
                                             'frequency', 
                                             'category',
                                             'condition'])        
    
    
    return df

