# Cyna Shirazinejad, 8/17/20, Drubin Lab

import os
import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import tkinter as tk
import imageio
import random
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import FloatSlider
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import warnings
import pickle
import scipy.stats as stats
import itertools
# add Python scripts to the local path

from display_tracks import *
# from maximum_intensity_analysis import *
from merge_tools import *
from alignment import *
from new_lifetime_functions import *
#from separate_tracking_utilities import *
from separate_tracking_merge_tools import *
from display_tracks import display_tracks_for_selection
import return_track_attributes

class TrackFeatures:
    """
    Extract track features from a Python-converted ProcessedTracks.mat or merged tracks items.
    
    Attributes:
        tracks (dictionary (with track indices as keys) or ndarray): tracks outputted by ProcessedTracks.mat
        current_features (list): features to be extracted or presently extracted from tracks
        feature_matrix (ndarray): matrix of features (n_samples x n_features)        
        feature_options (list): options for feature extraction from tracks
        significant_pval_cutoff (float, optional): p-val for accepting significant detections in secondary tracking channel(s)
        track_categories (list of ints, optional): categories of tracks, defined by cmeAnalysis, for feature extraction

    """
    
    def __init__(self, 
                 tracks, 
                 significant_pval_cutoff = 0.01,
                 track_categories = [1,2,3,4,5,6,7,8]):
        
        # the Python-converted tracked objects from cmeAnalysis' ProcessedTracks.mat
        self.tracks = tracks
        
        self.current_features = []
        
        self.track_categories = track_categories
        
        self.feature_matrix = []
        
        self.significant_pval_cutoff = significant_pval_cutoff
        
        self.feature_options = ['lifetime',
                                'max_int_ap2',
                                'max_int_dnm2',
                                'dist_traveled_ap2',
                                'dist_traveled_dnm2',
                                'max_dist_between_ap2_dnm2',
                                'md_ap2',
                                'md_dnm2',
                                'time_to_peak_ap2',
                                'time_to_peak_dnm2',
                                'time_after_peak_ap2',
                                'time_after_peak_dnm2',
                                'time_between_peaks_ap2_dnm2',
                                'avg_int_change_to_peak_ap2',
                                'avg_int_change_to_peak_dnm2',
                                'avg_int_change_after_peak_ap2',
                                'avg_int_change_after_peak_dnm2',
                                'peak_int_diff_ap2_dnm2',
                                'ratio_max_int_ap2_dnm2',
                                'mean_ap2',
                                'mean_dnm2',
                                'variation_ap2',
                                'variation_dnm2',
                                'skewness_ap2',
                                'skewness_dnm2',
                                'kurtosis_ap2',
                                'kurtosis_dnm2',
                                'number_significant_dnm2',
                                'max_consecutive_significant_dnm2',
                                'fraction_significant_dnm2',
                                'track_category']
        
        
    def add_features(self,
                     features):
        """Configure selected features from 'feature_options' to be extracted."""
        # add the desired features to be extracted for the data set 
        for feature in features:
            
            self.current_features.append(feature)
    
    def extract_features(self):
        """Generate a feature matrix (n_samples, n_features) based on designated features."""
        # re-initiate 'feature_matrix' as a blank list
        self.feature_matrix = []
        
        for feature in self.current_features:
            
            # call the function with name identical to the string options in 'feature_options'
            getattr(TrackFeatures, feature)(self)
        
        # reshape the list into a matrix, with N rows for every track, and M columns for every extracted feature
        self.feature_matrix = np.reshape(self.feature_matrix, (-1, len(self.current_features)), order='F')
            
    def lifetime(self):
        """Extracts raw lifetime (seconds)."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
                
                self.feature_matrix.append(return_track_attributes.return_track_lifetime(self.tracks, i))
        
    def max_int_ap2(self):
        """Extract maximum intensity (a.u. fluorescence) of first channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
                
                self.feature_matrix.append(np.max(return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)))
            
    def max_int_dnm2(self):
        """Extract maximum intensity (a.u. fluorescence) of second channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
                
                self.feature_matrix.append(np.max(return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)))
            
    def dist_traveled_ap2(self):
        """Extract distance (pixels) traveled from origin measured by localized centers of first channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                x_positions = return_track_attributes.return_puncta_x_position_no_buffer_one_channel(self.tracks, i, 0)
                y_positions = return_track_attributes.return_puncta_y_position_no_buffer_one_channel(self.tracks, i, 0)

                self.feature_matrix.append(np.sqrt((x_positions[-1]-x_positions[0])**2 + (y_positions[-1]-y_positions[0])**2))
            
    def dist_traveled_dnm2(self):
        """Extract distance (pixels) traveled from origin measured by localized centers of second channel."""
        for i,track in enumerate(self.tracks):

            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                x_positions = return_track_attributes.return_puncta_x_position_no_buffer_one_channel(self.tracks, i, 1)
                y_positions = return_track_attributes.return_puncta_y_position_no_buffer_one_channel(self.tracks, i, 1)

                self.feature_matrix.append(np.sqrt((x_positions[-1]-x_positions[0])**2 + (y_positions[-1]-y_positions[0])**2))
            
    def max_dist_between_ap2_dnm2(self):
        """Extract the maximum distance (pixels) between first and second channels measured by localized centers each channel."""
        for i,track in enumerate(self.tracks):

            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                x_positions_ap2 = return_track_attributes.return_puncta_x_position_no_buffer_one_channel(self.tracks, i, 0)
                y_positions_ap2 = return_track_attributes.return_puncta_y_position_no_buffer_one_channel(self.tracks, i, 0)

                x_positions_dnm2 = return_track_attributes.return_puncta_x_position_no_buffer_one_channel(self.tracks, i, 1)
                y_positions_dnm2 = return_track_attributes.return_puncta_y_position_no_buffer_one_channel(self.tracks, i, 1)

                distance_between_channels=[]

                for j in range(len(x_positions_ap2)):

                    distance_between_channels.append(np.sqrt((x_positions_ap2[j]-x_positions_dnm2[j])**2 + (y_positions_ap2[j]-y_positions_dnm2[j])**2))

                self.feature_matrix.append(np.max(distance_between_channels))
            
    def md_ap2(self):
        """Extract mean squared displacement (pixels) of first channel."""
        for i,track in enumerate(self.tracks):

            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                x_positions = return_track_attributes.return_puncta_x_position_no_buffer_one_channel(self.tracks, i, 0)
                y_positions = return_track_attributes.return_puncta_y_position_no_buffer_one_channel(self.tracks, i, 0)

                displacements=[]

                for j in range(1,len(x_positions)):

                    displacements.append((x_positions[j]-x_positions[j-1])**2 + (y_positions[j]-y_positions[j-1])**2)

                self.feature_matrix.append(np.sum(displacements)/len(displacements))

    def md_dnm2(self):
        """Extract mean squared displacement (pixels) of second channel."""
        for i,track in enumerate(self.tracks):

            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                x_positions = return_track_attributes.return_puncta_x_position_no_buffer_one_channel(self.tracks, i, 1)
                y_positions = return_track_attributes.return_puncta_y_position_no_buffer_one_channel(self.tracks, i, 1)

                displacements=[]

                for j in range(1,len(x_positions)):

                    displacements.append((x_positions[j]-x_positions[j-1])**2 + (y_positions[j]-y_positions[j-1])**2)

                self.feature_matrix.append(np.sum(displacements)/len(displacements))
            
    def time_to_peak_ap2(self):
        """Extract the time (seconds) for the intensity of first channel to reach peak value; 0 if peak at first frame."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)

                if np.argmax(ap2_amplitudes) == 0:

                    self.feature_matrix.append(0)

                else:

                    self.feature_matrix.append(np.argmax(ap2_amplitudes))
            
    def time_to_peak_dnm2(self):
        """Extract the time (seconds) for the intensity of second channel to reach peak value; 0 if peak at first frame."""                                  
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)

                if np.argmax(dnm2_amplitudes) == 0:

                    self.feature_matrix.append(0)

                else:

                    self.feature_matrix.append(np.argmax(dnm2_amplitudes))
                                           
    def time_after_peak_ap2(self):
        """Extract the time (seconds) after the intensity of first channel reaches its peak value; 0 if peak at last frame."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)

                if np.argmax(ap2_amplitudes) == (len(ap2_amplitudes) - 1):

                    self.feature_matrix.append(0)

                else:

                    self.feature_matrix.append(len(ap2_amplitudes) - 1 - np.argmax(ap2_amplitudes))
            
    def time_after_peak_dnm2(self):
        """Extract the time (seconds) after the intensity of second channel reaches its peak value; 0 if peak at last frame."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
                
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)

                if np.argmax(dnm2_amplitudes) == (len(dnm2_amplitudes) - 1):

                    self.feature_matrix.append(0)

                else:

                    self.feature_matrix.append(len(dnm2_amplitudes) - 1 - np.argmax(dnm2_amplitudes))            

    def time_between_peaks_ap2_dnm2(self):
        """Extract the time (seconds) between the peaks of the first and second channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)                            
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)

                self.feature_matrix.append(np.argmax(ap2_amplitudes) - np.argmax(dnm2_amplitudes))

                        
    def avg_int_change_to_peak_ap2(self):
        """Extract the average change in the first channel's intensity (a.u. fluorescence) to the peak; 0 if peak is at first frame."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                change = 0
                num_changes = 0                               
                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)    

                if np.argmax(ap2_amplitudes) != 0:

                    for frame in range(np.argmax(ap2_amplitudes)):

                        change += (ap2_amplitudes[frame+1] - ap2_amplitudes[frame])
                        num_changes+=1
                    self.feature_matrix.append(change/num_changes)

                else:

                    self.feature_matrix.append(0)

    def avg_int_change_to_peak_dnm2(self):
        """Extract the average change in the second channel's intensity (a.u. fluorescence) to the peak; 0 if peak is at first frame."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                change = 0
                num_changes = 0                              
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)    

                if np.argmax(dnm2_amplitudes) != 0:

                    for frame in range(np.argmax(dnm2_amplitudes)):

                        change += (dnm2_amplitudes[frame+1] - dnm2_amplitudes[frame])
                        num_changes+=1
                    self.feature_matrix.append(change/num_changes)

                else:

                    self.feature_matrix.append(0)
            
    def avg_int_change_after_peak_ap2(self):
        """Extract the average change in the first channel's intensity (a.u. fluorescence) after the peak; 0 if peak is at last frame."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
                
                change = 0
                num_changes = 0                               
                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)    

                if np.argmax(ap2_amplitudes) != (len(ap2_amplitudes) - 1):

                    for frame in range(np.argmax(ap2_amplitudes),len(ap2_amplitudes)-1):

                        change += (ap2_amplitudes[frame+1] - ap2_amplitudes[frame])
                        num_changes+=1
                    self.feature_matrix.append(change/num_changes)

                else:

                    self.feature_matrix.append(0)
            
    def avg_int_change_after_peak_dnm2(self):
        """Extract the average change in the second channel's intensity (a.u. fluorescence) after the peak; 0 if peak is at last frame."""
        for i,track in enumerate(self.tracks):

            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                change = 0
                num_changes = 0                           
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)    

                if np.argmax(dnm2_amplitudes) != (len(dnm2_amplitudes) - 1):

                    for frame in range(np.argmax(dnm2_amplitudes),len(dnm2_amplitudes)-1):

                        change += (dnm2_amplitudes[frame+1] - dnm2_amplitudes[frame])
                        num_changes+=1
                    self.feature_matrix.append(change/num_changes)

                else:

                    self.feature_matrix.append(0)
            
    def peak_int_diff_ap2_dnm2(self):
        """Extract the difference between the peak intensities (a.u. fluorescence) of the first and second channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)  
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)  

                self.feature_matrix.append(ap2_amplitudes[np.argmax(ap2_amplitudes)] - dnm2_amplitudes[np.argmax(dnm2_amplitudes)])
            
    def ratio_max_int_ap2_dnm2(self):
        """Extract the ratio between the peak intensities (a.u. fluorescence) of the first and second channel"""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)  
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)  

                self.feature_matrix.append(ap2_amplitudes[np.argmax(ap2_amplitudes)] / dnm2_amplitudes[np.argmax(dnm2_amplitudes)])
            
    def mean_ap2(self):
        """Extract the mean intensity (a.u. fluorescence) of the first channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:                               
                
                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)  

                self.feature_matrix.append(np.mean(ap2_amplitudes))
                                           
    def mean_dnm2(self):
        """Extract the mean intensity (a.u. fluorescence) of the second channel"""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:                              
                
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)  

                self.feature_matrix.append(np.mean(dnm2_amplitudes))
            
    def variation_ap2(self):
        """Extract the variation in intensity (a.u. fluorescence^2) of the first channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:                              
                
                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)  

                self.feature_matrix.append(stats.variation(ap2_amplitudes))
            
    def variation_dnm2(self):
        """Extract the variation in intensity (a.u. fluorescence^2) of the second channel."""                                   
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:                              
                
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)  

                self.feature_matrix.append(stats.variation(dnm2_amplitudes))

    def skewness_ap2(self):
        """Extract the skewness in intensity (unitless) of the first channel."""                                  
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:                              
                
                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)  

                self.feature_matrix.append(stats.skew(ap2_amplitudes))
            
    def skewness_dnm2(self):
        """Extract the skewness in intensity (unitless) of the second channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:                             
                
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)  

                self.feature_matrix.append(stats.skew(dnm2_amplitudes))

    def kurtosis_ap2(self):
        """Extract the kurtosis in intensity (unitless) of the first channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:

                ap2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 0)  

                self.feature_matrix.append(stats.kurtosis(ap2_amplitudes))
            
    def kurtosis_dnm2(self):
        """Extract the kurtosis in intensity (unitless) of the second channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:  
                
                dnm2_amplitudes = return_track_attributes.return_track_amplitude_no_buffer_channel(self.tracks, i, 1)  

                self.feature_matrix.append(stats.kurtosis(dnm2_amplitudes))
    
    def number_significant_dnm2(self):
        """Extract the number (counts) of significant detections below the designated p-value in the second channel."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
                
                pvals = return_track_attributes.return_pvals_detection_no_buffer(self.tracks,
                                               i,
                                               1)

                self.feature_matrix.append(len(np.where(pvals<self.significant_pval_cutoff)[0]))
    
    def max_consecutive_significant_dnm2(self):
        """Extract the maximum number (counts) of consecutive significant detections below the designated p-value in the second channel."""        
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
                
                pvals = return_track_attributes.return_pvals_detection_no_buffer(self.tracks,
                                               i,
                                               1)

                significant_pval_indices = [1 if pval < self.significant_pval_cutoff else 0 for pval in pvals]

                repeated_indices = [(x[0], len(list(x[1]))) for x in itertools.groupby(significant_pval_indices)]

                max_1s = 0

                for itm in repeated_indices:
                    if itm[0] == 1:
                        if itm[1]>max_1s:
                            max_1s=itm[1]

                self.feature_matrix.append(max_1s)
            
    def fraction_significant_dnm2(self):
        """Extract the fraction (unitless) of the second channel's lifetime that contains significant detections below the designated p-value."""
        for i,track in enumerate(self.tracks):
            
            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
                
                pvals = return_track_attributes.return_pvals_detection_no_buffer(self.tracks,
                                           i,
                                           1)
            
                self.feature_matrix.append((pvals < self.significant_pval_cutoff).sum()/len(pvals))
        
    def track_category(self):
        """Extract the category of the track."""
        for i,track in enumerate(self.tracks):

            if return_track_attributes.return_track_category(self.tracks,i) in self.track_categories:
            
                self.feature_matrix.append(return_track_attributes.return_track_category(self.tracks,i))
