import scipy.io as sio
from generate_index_dictionary import return_index_dictionary
import feature_extraction_modular
import feature_extraction_with_buffer

import generate_index_dictionary

index_dictionary = return_index_dictionary()
import os
import numpy as np
import merge_tools
import pandas as pd
from return_track_attributes import (return_track_category, 
                                     return_track_lifetime,
                                     return_track_amplitude,
                                     return_track_amplitude_one_channel)
def load_tracks(track_path):
    """Return a Python structure of the tracks, sorted from decreasing lifetime, from the ProcessedTracks.mat output of cmeAnalysis, with keys designated track indices"""
    return sort_tracks_descending_lifetimes(sio.loadmat(track_path)) # sort the tracks by descending lifetime order

def sort_tracks_descending_lifetimes(tracks):
    """Sort tracks in descending lifetime order"""
    index_dictionary = return_index_dictionary()
    
    tracks = tracks['tracks'][0] # get just the data for the tracks
                                            
    tracks = zip(tracks, range(len(tracks))) # save each track with its original index
    # sort the tracks in descending lifetime, preserving each track's individual index
    tracks = sorted(tracks, key=lambda track: track[0][index_dictionary['index_lifetime_s']][0][0], reverse=True) 
    
    (tracks, indices) = zip(*tracks) 
    
    return tracks

def upload_tracks_and_metadata(path_tracks,
                               path_outputs,
                               analysis_metadata,
                               track_categories,
                               identifier_strings,
                               features,
                               labels,
                               dataframe_name,
                               track_name):
    """
    Format tracks contained in folders into a dataframe of extracted physical features 
    as well as accompanying experimental metadata.
    
    Args:
        analysis_metadata (dictionary): contains path to the folder containing the enclosed tracking files and output files
        track_categories (list): a list of integers for the cmeAnalysis track categories to keep for further analysis
        identifier_string (string): a label within all tracking head folders that uniquely identify relevant content
        features (list): a list with string elements containing the features to be extracted from each track
        labels (list): a list of string elements describing each features' designation in the dataframe output
        experiment_number_adjustment (int): a number to offset the starting cout for experiment number, nonzero if data is to be 
            appended to another existing dataset
    
    Returns:
    
        df (dataframe): a dataframe of features and metadata
        merged_all_tracks (ndarray): 
    """
    temp_paths = os.listdir(path_tracks)
    all_track_paths = []
    for exp in temp_paths:
        num_matches = 0
        for identifier in identifier_strings:
            if identifier in exp:
                num_matches+=1
        if num_matches==len(identifier_strings):
            all_track_paths.append(exp)
            
    exp_num_index = [int(exp.split('_')[0]) for exp in all_track_paths]
    all_track_paths = [all_track_paths[idx] for idx in np.argsort(exp_num_index)]
        
    print('\nfolders to mine:')
    for exp_name in all_track_paths:    
        print(exp_name)
    print('\n')
    
    tracks = []
    dates = []
    cell_line_tags = []
    current_tracked_channels = []
    number_of_tags = []
    experiment = []
    condition = []
    experiment_number = []
    framerates = []
    
    print('uploading and saving tracks...\n')
    
    for exp_number, exp in enumerate(all_track_paths):
        
        current_tracks = load_tracks(path_tracks + '/' + exp + '/Ch1/Tracking/ProcessedTracks.mat')
        current_tracks = remove_tracks_by_criteria(current_tracks, track_category=track_categories)
        tracks.append(current_tracks)
        
        np.save(path_outputs+"/dataframes/"+track_name+"_"+str(exp_number), np.array(list(current_tracks)))
        
        num_tracks = len(current_tracks)
        
        metadata = exp.split('_')
        
        experiment_number += [int(metadata[0])]*num_tracks
        dates += [int(metadata[1])]*num_tracks
        cell_line_tags += [metadata[2]]*num_tracks
        current_tracked_channels += [metadata[3]]*num_tracks
        number_of_tags += [len(metadata[3].split('-'))]*num_tracks
        experiment += [metadata[4]]*num_tracks
        condition += [metadata[5]]*num_tracks
        framerates += [metadata[-1]]*num_tracks
        
    print('\nfinished uploading tracks\n')
    merged_all_tracks = merge_tools.merge_experiments(tracks,[list(range(len(track_set))) for track_set in tracks])
    
    # extract the output of cmeAnalysis' predictions on whether a track is DNM2 positive or negative
    significant_dynamin2_cmeAnalysis_prediction = []

    # an index map for ProcessedTracks.mat attributes for 2 color tracking experiments from cmeAnalysis
    index_dictionary = generate_index_dictionary.return_index_dictionary()
    
    for track in merged_all_tracks: # iterate through all tracks
#         print(track)
        significant_dynamin2 = track[index_dictionary['index_significantSlave']][1]
        significant_dynamin2_cmeAnalysis_prediction.append(significant_dynamin2)
    print('extracting features...\n')
    all_track_features = feature_extraction_modular.TrackFeatures(merged_all_tracks) # an instance of a to-be feature matrix of tracks
    all_track_features.add_features(features) # set the features to be extracted
    all_track_features.extract_features() # extract all features
    extracted_features = all_track_features.feature_matrix # feature matrix for all tracks
    print('completed feature extraction\n')
    # merge features with labels (experiment number, date, and number of channels)
    extracted_features = np.array(extracted_features)
    merged_features = np.concatenate((extracted_features,
                                      np.array(significant_dynamin2_cmeAnalysis_prediction).reshape(extracted_features.shape[0],-1)), axis=-1)
    merged_features = np.concatenate((merged_features,
                                      np.array(experiment_number).reshape(merged_features.shape[0],-1)), axis=-1)
    merged_features = np.concatenate((merged_features,
                                      np.array(number_of_tags).reshape(merged_features.shape[0],-1)), axis=-1)
    merged_features = np.concatenate((merged_features,
                                      np.array(cell_line_tags).reshape(merged_features.shape[0],-1)), axis=-1)
    merged_features = np.concatenate((merged_features,
                                      np.array(current_tracked_channels).reshape(merged_features.shape[0],-1)), axis=-1)
    merged_features = np.concatenate((merged_features,
                                      np.array(experiment).reshape(merged_features.shape[0],-1)), axis=-1)
    merged_features = np.concatenate((merged_features,
                                      np.array(condition).reshape(merged_features.shape[0],-1)), axis=-1)
    merged_features = np.concatenate((merged_features,
                                      np.array(framerates).reshape(merged_features.shape[0],-1)), axis=-1)
    merged_features = np.concatenate((merged_features,
                                      np.array(dates).reshape(merged_features.shape[0],-1)), axis=-1)
    
    print('creating dataframe...\n')

    df_merged_features = pd.DataFrame(data=merged_features, columns=labels+['cmeAnalysis_dynamin2_prediction',
                                                            'experiment_number',
                                                            'number_of_tags', 
                                                            'cell_line_tags',
                                                            'current_tracked_channels',
                                                            'experiment_type', 
                                                            'cell_condition', 
                                                            'framerate',
                                                            'date'])
    
    print('saving dataframe...\n')
    # save the dataframe for subsequent notebooks
    compression_opts = dict(method='zip',
                            archive_name=path_outputs+'/dataframes/'+dataframe_name+'.csv')  

    df_merged_features.to_csv(path_outputs+'/dataframes/'+dataframe_name+'.zip', index=False,
                                                              compression=compression_opts) 

    
    number_of_track_splits = len(tracks)
    
    analysis_metadata.item()['number_of_track_splits_'+track_name] = number_of_track_splits

    np.save(path_outputs+'/dataframes/analysis_metadata', analysis_metadata)
    
    print('saving tracks...\n')
    # split tracks
#     split_valid_tracks = np.array_split(np.array(list(merged_all_tracks)),number_of_track_splits)
    # save each track array chunk
    for i in range(len(tracks)):

        np.save(path_outputs+"/dataframes/"+track_name+"_"+str(i), np.array(list(tracks[i])))
        
    print('done')
    
    return df_merged_features, merged_all_tracks

# def upload_tracks_and_metadata_with_buffer(path_tracks,
#                                path_outputs,
#                                analysis_metadata,
#                                track_categories,
#                                identifier_string,
#                                features,
#                                labels,
#                                dataframe_name,
#                                track_name,
#                                experiment_number_adjustment=0):
#     """
#     Format tracks contained in folders into a dataframe of extracted physical features 
#     as well as accompanying experimental metadata.
    
#     Args:
#         analysis_metadata (dictionary): contains path to the folder containing the enclosed tracking files and output files
#         track_categories (list): a list of integers for the cmeAnalysis track categories to keep for further analysis
#         identifier_string (string): a label within all tracking head folders that uniquely identify relevant content
#         features (list): a list with string elements containing the features to be extracted from each track
#         labels (list): a list of string elements describing each features' designation in the dataframe output
#         experiment_number_adjustment (int): a number to offset the starting cout for experiment number, nonzero if data is to be 
#             appended to another existing dataset
    
#     Returns:
    
#         df (dataframe): a dataframe of features and metadata
#         merged_all_tracks (ndarray): 
#     """
#     all_track_paths = os.listdir(path_tracks)
#     all_track_paths = [exp for exp in all_track_paths if identifier_string in exp]
#     all_track_paths.sort()
#     print('\nfolders to mine:')
#     for exp_name in all_track_paths:    
#         print(exp_name)
#     print('\n')
    
#     tracks = []
#     dates = []
#     cell_line_tags = []
#     current_tracked_channels = []
#     number_of_tags = []
#     experiment = []
#     condition = []
#     experiment_number = []
#     framerates = []
#     print('uploading and saving tracks...\n')
#     for exp_number, exp in enumerate(all_track_paths):
        
#         current_tracks = load_tracks(path_tracks + '/' + exp + '/Ch1/Tracking/ProcessedTracks.mat')
#         current_tracks = remove_tracks_by_criteria(current_tracks, track_category=track_categories)
#         tracks.append(current_tracks)
        
#         np.save(path_outputs+"/dataframes/"+track_name+"_"+str(exp_number), np.array(current_tracks))
        
#         num_tracks = len(current_tracks)
        
#         metadata = exp.split('_')
        
# #         tracks += current_tracks
#         dates += [int(metadata[0])]*num_tracks
#         cell_line_tags += [metadata[1]]*num_tracks
#         current_tracked_channels += [metadata[2]]*num_tracks
#         number_of_tags += [len(metadata[1].split('-'))]*num_tracks
#         experiment += [metadata[3]]*num_tracks
#         condition += [metadata[4]]*num_tracks
#         experiment_number += [exp_number+experiment_number_adjustment]*num_tracks
#         framerates += [metadata[6]]*num_tracks
        
#     print('\nfinished uploading tracks\n')
#     merged_all_tracks = merge_tools.merge_experiments(tracks,[list(range(len(track_set))) for track_set in tracks])
    
#     # extract the output of cmeAnalysis' predictions on whether a track is DNM2 positive or negative
#     significant_dynamin2_cmeAnalysis_prediction = []

#     # an index map for ProcessedTracks.mat attributes for 2 color tracking experiments from cmeAnalysis
#     index_dictionary = generate_index_dictionary.return_index_dictionary()
    
#     for track in merged_all_tracks: # iterate through all tracks
# #         print(track)
#         significant_dynamin2 = track[index_dictionary['index_significantSlave']][1]
#         significant_dynamin2_cmeAnalysis_prediction.append(significant_dynamin2)
#     print('extracting features...\n')
#     all_track_features = feature_extraction_with_buffer.TrackFeatures(merged_all_tracks) # an instance of a to-be feature matrix of tracks
#     all_track_features.add_features(features) # set the features to be extracted
#     all_track_features.extract_features() # extract all features
#     extracted_features = all_track_features.feature_matrix # feature matrix for all tracks
#     print('completed feature extraction\n')
#     # merge features with labels (experiment number, date, and number of channels)
#     extracted_features = np.array(extracted_features)
#     merged_features = np.concatenate((extracted_features,
#                                       np.array(significant_dynamin2_cmeAnalysis_prediction).reshape(extracted_features.shape[0],-1)), axis=-1)
#     merged_features = np.concatenate((merged_features,
#                                       np.array(experiment_number).reshape(merged_features.shape[0],-1)), axis=-1)
#     merged_features = np.concatenate((merged_features,
#                                       np.array(number_of_tags).reshape(merged_features.shape[0],-1)), axis=-1)
#     merged_features = np.concatenate((merged_features,
#                                       np.array(cell_line_tags).reshape(merged_features.shape[0],-1)), axis=-1)
#     merged_features = np.concatenate((merged_features,
#                                       np.array(current_tracked_channels).reshape(merged_features.shape[0],-1)), axis=-1)
#     merged_features = np.concatenate((merged_features,
#                                       np.array(experiment).reshape(merged_features.shape[0],-1)), axis=-1)
#     merged_features = np.concatenate((merged_features,
#                                       np.array(condition).reshape(merged_features.shape[0],-1)), axis=-1)
#     merged_features = np.concatenate((merged_features,
#                                       np.array(framerates).reshape(merged_features.shape[0],-1)), axis=-1)
#     merged_features = np.concatenate((merged_features,
#                                       np.array(dates).reshape(merged_features.shape[0],-1)), axis=-1)
    
#     print('creating dataframe...\n')

#     df_merged_features = pd.DataFrame(data=merged_features, columns=labels+['cmeAnalysis_dynamin2_prediction',
#                                                             'experiment_number',
#                                                             'number_of_tags', 
#                                                             'cell_line_tags',
#                                                             'current_tracked_channels',
#                                                             'experiment_type', 
#                                                             'cell_condition', 
#                                                             'framerate',
#                                                             'date'])
    
#     print('saving dataframe...\n')
#     # save the dataframe for subsequent notebooks
#     compression_opts = dict(method='zip',
#                             archive_name=path_outputs+'/dataframes/'+dataframe_name+'.csv')  

#     df_merged_features.to_csv(path_outputs+'/dataframes/'+dataframe_name+'.zip', index=False,
#                                                               compression=compression_opts) 

    
# #     number_of_track_splits = 20
    
# #     analysis_metadata.item()['number_of_track_splits'] = number_of_track_splits

# #     np.save(path_outputs+'/dataframes/analysis_metadata', analysis_metadata)
    
# #     print('saving tracks...\n')
#     # split tracks
# #     split_valid_tracks = np.array_split(np.array(list(merged_all_tracks)),number_of_track_splits)
#     # save each track array chunk
# #     for i in range(len(split_valid_tracks)):

# #         np.save(path_outputs+"/dataframes/"+track_name+"_"+str(i), np.array(split_valid_tracks[i]))
        
# #     print('done')
    
#     return df_merged_features, merged_all_tracks

def remove_tracks_by_criteria(tracks, 
                              minimum_lifetime=0, 
                              maximum_lifetime=np.Inf, 
                              track_category=[1, 2, 3, 4, 5, 6, 7, 8], 
                              ch_min_int_threshold=[-np.Inf], 
                              keep_all=False, 
                              only_ccps=False, 
                              constraints_overlap_ccps=False):    
    tracks_return = []
    new_index = 0
    for i in range(0, len(tracks)):

        if keep_all:
            

            tracks_return.append(tracks[i])
    
        elif only_ccps:

            if return_is_CCP(tracks, i) == 1:

                tracks_return.append(tracks[i])
                
        elif constraints_overlap_ccps:

            if track_within_bounds(tracks, i, minimum_lifetime, maximum_lifetime, track_category) and \
               return_is_CCP(tracks, i) == 1:
                
                tracks_return.append(tracks[i])



        elif track_within_bounds(tracks, i, minimum_lifetime, maximum_lifetime, track_category):

            num_channels_in_track = len(return_track_amplitude(tracks, i))


            if len(ch_min_int_threshold) == 1:
            
                temp = 0

                for j in range(num_channels_in_track):

                    if max(return_track_amplitude_one_channel(tracks, i, j)) > ch_min_int_threshold[0]:
                        
                        temp += 1

                if temp == num_channels_in_track:

                    tracks_return.append(tracks[i])
    
            elif len(ch_min_int_threshold) == num_channels_in_track:

                temp = 0

                for j in range(num_channels_in_track):

                    if max(return_track_amplitude_one_channel(tracks, i, j)) > ch_min_int_threshold[j]:

                        temp += 1

                if temp == num_channels_in_track:


                    tracks_return.append(tracks[i])
                    
#     print('The number of tracks returned: ' + str(len(tracks_return)))
#     print()
    return tuple(tracks_return)

def track_within_bounds(tracks, track_number, minimum_lifetime, maximum_lifetime, track_category):
    """Check if a track satisfies the user-defined conditions"""
    if return_track_category(tracks, track_number) in track_category and \
       return_track_lifetime(tracks, track_number) >= minimum_lifetime and \
       return_track_lifetime(tracks, track_number) <= maximum_lifetime:

        return True

    else:

        return False
