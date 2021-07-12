# Cyna Shirazinejad, last modified 10/7/20
# utilities to build KDTrees from tracking data

import os
import sys
sys.path.append(os.getcwd())
from generate_index_dictionary import return_index_dictionary
import numpy as np
from sklearn.neighbors import KDTree
from collections import Counter
from scipy.stats import mode
import scipy.stats as stats
from return_track_attributes import (return_track_category, return_track_lifetime, return_track_amplitude, 
                                     return_track_x_position, return_is_CCP, return_track_amplitude_one_channel,
                                     return_puncta_x_position_no_buffer_one_channel, 
                                     return_puncta_y_position_no_buffer_one_channel,
                                     return_distance_traveled_from_origin, return_frames_in_track_no_buffer,
                                     return_distance_between_two_channel)




def associate_tracks(primary_track_set,
                     primary_channel,
                     candidate_neighbor_track_set,
                     second_channel,
                     number_of_frames,
                     search_radius=3):
    """
    Generate a list of candidate associated tracks to link separately tracked channels of a movie.
    Each track will have a list of nearest neighbors for each frame the track is in.
    
    Args:
        primary_track_set (dictionary (with track indices as keys) or ndarray): tracks containing the channel to search assign neighbors to
        primary_channel (int): imaging channel in primary_track_set to find neighbors for
        candidate_neighbor_track_set (dictionary (with track indices as keys) or ndarray): tracks containing the channel to build KDTrees from
        second_channel (int): imaging channel in candidate_neighbor_track_set to build KDTrees from
        number_of_frames (int): number of frames in movie
        search_radius (int, optional): radius (in pixels) of querying for nearest neighbors in KDTree
        
    Returns:
        associated_tracks (list): element are lists of the corresponding indexed track containing nearest neighbor indices in secondary channel 
        distances_per_track (list): elements are lists corresponding to the nearest neighbors of each track for every one of its frames
        
    Note:
        All return values contain elements of -1 if the primary channel has no neighbor within search radius in the secondary channel.
        
    """
    trees_candidate_neighbors, vals_from_tree_candidate_neighbors = build_kd_tree_channel(candidate_neighbor_track_set,
                                                                                          second_channel,
                                                                                          number_of_frames)
    
    return associate_tracks_from_trees(primary_track_set,
                                       primary_channel,
                                       trees_candidate_neighbors,
                                       vals_from_tree_candidate_neighbors,
                                       search_radius)

def build_kd_tree_channel(candidate_neighbor_track_set,
                          second_channel,
                          len_movie):
    """
    Construct KDTrees for every frame of a movie. The coordinates of the tree contain the fitted x,y positions
    of tracks who belong in each frame.
    
    Args:
        candidate_neighbor_track_set (dictionary (with track indices as keys) or ndarray): tracks to build trees from
        second_channel (int): imaging channel in candidate_neighbor_track_set to construct trees from
        len_movie (int): length of the movie 
        
    Returns:
        trees_per_frame (list): each element is a KDTree in order of the frames of the movie
        tree_vals (ndarray): each row of the array contains the track index, track x position, and track y position
    """
    tracks = candidate_neighbor_track_set
    
    track_positions = []
    
    for i in range(len(tracks)): # iterate through all tracks

        track_x_positions = return_puncta_x_position_no_buffer_one_channel(tracks,i,second_channel) # extract positions
        track_y_positions = return_puncta_y_position_no_buffer_one_channel(tracks,i,second_channel)
        frames = return_frames_in_track_no_buffer(tracks,i)-1 # extract the frames in the movie the track belongs in
       
        # each entry for tree_vals contains the track index, track x position, and track y position
        track_positions.append(list(zip([i for j in range(len(track_x_positions))],frames,track_x_positions, track_y_positions)))

    
    # each index i contains a tree built from all puncta in frame i
    trees_per_frame=[]
    
    # each index i contains a matrix (nx3) of [track_ID, x, y] for n puncta present in frame i
    tree_vals = []
    
    for frame in range(len_movie): # iterate through all frames in the movie to check which tracks belong to each frame

        current_tree_vals = []
#         
        for track in track_positions:

            for time_point in track:
                
                # check whether frame contains track 
                if frame == time_point[1]:
                    
                    # check for erronous nan fits of x and y 
                    if time_point[2] < np.Inf and time_point[3] < np.Inf:
                        
                        # adding a list [track_ID, x, y]
                        current_tree_vals.append((time_point[0],time_point[2],time_point[3]))
                        
                    else:
                        
                        print('nan position found')
                        
        if current_tree_vals!=[]: # if the frame does contain tracks in it, construct a tree
  
            current_tree_vals=np.array(current_tree_vals)
        
            tree_vals.append(current_tree_vals)
        
            # form a tree just from x y positions (excluding track idendity)
            current_tree = KDTree(current_tree_vals[:,1:3])               

            trees_per_frame.append(current_tree)
    
        else:
            
            tree_vals.append(-1)
            trees_per_frame.append(-1)
        
    return trees_per_frame, tree_vals




def associate_tracks_from_trees(primary_track_set,
                                primary_channel,
                                trees_candidate_neighbors,
                                vals_from_tree_candidate_neighbors,
                                search_radius):
    """
    Associate tracks from the primary channel to the secondary channel as searched from KDTrees of every frame.
    
    Args:
        primary_track_set (dictionary (with track indices as keys) or ndarray): tracks containing the channel to search assign neighbors to
        primary_channel (int): imaging channel in primary_track_set to find neighbors for
        trees_candidate_neighbors (list): each element is a KDTree in order of the frames of the movie
        vals_from_tree_candidate_neighbors (ndarray): each row of the array contains the track index, track x position, and track y position
        search_radius (int, optional): radius (in pixels) of querying for nearest neighbors in KDTree

    Returns:
        associated_tracks (list): element are lists of the corresponding indexed track containing nearest neighbor indices in secondary channel 
        distances_per_track (list): elements are lists corresponding to the nearest neighbors of each track for every one of its frames
    """
    # data to keep track of
    associated_tracks = [] # list of lists containing nearest neighbor. i'th list contains i'th primary track's nearest neighbors in everyone one of its frames
    frac_associated=[] 
    distances=[]
    distances_per_track = []
    distance_mode_all = []
    distance_mode_per_track = []
    
    tracks = primary_track_set
    kd_trees = trees_candidate_neighbors
    vals_tree = vals_from_tree_candidate_neighbors
    distance_query = search_radius
    
    for i in range(len(tracks)): # iterate through all primary channel tracks
         
        frames = return_frames_in_track_no_buffer(tracks,i)-1 # extract frames from track

        x_positions = return_puncta_x_position_no_buffer_one_channel(tracks,i,primary_channel) # extract fitted positions of track
        y_positions = return_puncta_y_position_no_buffer_one_channel(tracks,i,primary_channel)

        dist_individual_track = [] # distance to secondary channel nearest neighbor at every frame
        current_track_positions = list(zip(x_positions,y_positions))
        track_associated = []
        
        for j,frame in enumerate(frames):
            
            current_tree = kd_trees[int(frame)] # extract KDTree at the j'th frame
                
            if current_tree != -1: # if tree exists in secondary channel at this frame
                
                # find the nearest neighbor
                ind,dist = current_tree.query_radius(np.array(current_track_positions[j]).reshape(1,-1),
                                                     r=distance_query,
                                                     return_distance=True,
                                                     sort_results=True)
                

                if ind[0].size>0: # if there are neighbors

                    track_associated.append(int(vals_tree[int(frame)][ind[0][0]][0])) # find the index of the secondary channel nearest neighbor
                    distances.append(dist[0][0])
                    dist_individual_track.append(dist[0][0])

                else:

                    track_associated.append(-1)
                    distances.append(-1)
                    dist_individual_track.append(-1)
                    
            else:

                track_associated.append(-1)
                distances.append(-1)
                dist_individual_track.append(-1) 
                
        distances_per_track.append(dist_individual_track)
        associated_tracks.append(track_associated)
        mode = stats.mode(track_associated) # calculate distances and frequency of association with the mode nearest neighbor

        indices_mode = [i for i, x in enumerate(track_associated) if x == mode[0][0]] 

        distance_mode_per_track.append(np.array(dist_individual_track)[indices_mode])
        
        for dist in np.array(dist_individual_track)[indices_mode]:
            
            distance_mode_all.append(dist)
            
        num_associations=mode[1][0]
        frac_associated.append(num_associations/len(frames))
       
    return associated_tracks, distances_per_track
#     return associated_tracks, frac_associated, distances, distances_per_track, distance_mode_all, distance_mode_per_track




