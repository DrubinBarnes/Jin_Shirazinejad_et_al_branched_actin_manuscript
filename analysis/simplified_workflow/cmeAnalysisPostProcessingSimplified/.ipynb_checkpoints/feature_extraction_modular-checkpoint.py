# Cyna Shirazinejad, 1/14/22, Drubin Lab

import os
import sys
sys.path.append(os.getcwd())
import scipy.stats as stats
import itertools
import numpy as np
# add Python scripts to the local path
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
                 framerate=1,
                 buffer='on',
                 significant_pval_cutoff = 0.01):
        
        # the Python-converted tracked objects from cmeAnalysis' ProcessedTracks.mat
        self.tracks = tracks
        
        self.current_features = []
                
        self.feature_matrix = []
        
        self.significant_pval_cutoff = significant_pval_cutoff
        
        self.framerate = framerate
        
        self.feature_options = ['lifetime',
                                'max_int_chA',
                                'dist_traveled_chA',
                                'max_dist_between_chA-B',
                                'md_chA',
                                'time_to_peak_chA',
                                'time_after_peak_chA',
                                'time_between_peaks_chA-B',
                                'avg_int_change_to_peak_chA',
                                'avg_int_change_after_peak_chA',
                                'peak_int_diff_chA-B',
                                'ratio_max_int_chA-B',
                                'mean_chA',
                                'variation_chA',
                                'skewness_chA',
                                'kurtosis_chA',
                                'number_significant_chA',
                                'max_consecutive_significant_chA',
                                'fraction_significant_chA',
                                'fraction_peak_chA']
        
        
        
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
            
            # call the function with the designated feature

            if 'ch' in feature.split('_')[-1]:
                
                split_feature = feature.split('_')
                channels = split_feature[-1]
                
                # if there is more than one channel involved:
                if len(channels.split('-'))>1:
                    
                    print('placeholder')
                
                else:
                    
                    feature_function = ''
                    
                    for feature_name_fragment in split_feature[:-1]:
                        
                        feature_function += feature_name_fragment + '_'
                    
                    feature_function += 'chA'
                            
                    channel = int(channels[-1])
                    getattr(TrackFeatures, feature_function)(self, channel)
            else:
                
                getattr(TrackFeatures, feature)(self)
        
        # reshape the list into a matrix, with N rows for every track, and M columns for every extracted feature
        self.feature_matrix = np.reshape(self.feature_matrix, (-1, len(self.current_features)), order='F')
            
    def lifetime(self):
        """Extracts raw lifetime (seconds)."""
        for i,track in enumerate(self.tracks):
            
            self.feature_matrix.append(return_track_attributes.return_track_lifetime(self.tracks, i))
        
    def max_int_chA(self, channel):
        """Extract maximum intensity (a.u. fluorescence) of channel."""
        for i,track in enumerate(self.tracks):
            
            self.feature_matrix.append(np.max(return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)))
            

    def dist_traveled_chA(self, channel):
        """Extract distance (pixels) traveled from origin measured by localized centers of channel."""
        for i,track in enumerate(self.tracks):
            
            x_positions = return_track_attributes.return_puncta_x_position_whole_track(self.tracks, i, channel)
            y_positions = return_track_attributes.return_puncta_y_position_whole_track(self.tracks, i, channel)

            self.feature_matrix.append(np.sqrt((x_positions[-1]-x_positions[0])**2 + (y_positions[-1]-y_positions[0])**2))
            

    def max_dist_between_chA_B(self, channels):
        """Extract the maximum distance (pixels) between first and second channels measured by localized centers channel."""
        for i,track in enumerate(self.tracks):


                x_positions_chA = return_track_attributes.return_puncta_x_position_whole_track(self.tracks, i, channels[0])
                y_positions_chA = return_track_attributes.return_puncta_y_position_whole_track(self.tracks, i, channels[0])

                x_positions_chB = return_track_attributes.return_puncta_x_position_whole_track(self.tracks, i, channels[1])
                y_positions_chB = return_track_attributes.return_puncta_y_position_whole_track(self.tracks, i, channels[1])

                distance_between_channels=[]

                for j in range(len(x_positions_chA)):

                    distance_between_channels.append(np.sqrt((x_positions_chA[j]-x_positions_chB[j])**2 + (y_positions_chA[j]-y_positions_chB[j])**2))

                self.feature_matrix.append(np.max(distance_between_channels))
            
    def md_chA(self, channel):
        """Extract mean displacement (pixels) of channel."""
        for i,track in enumerate(self.tracks):

            x_positions = return_track_attributes.return_puncta_x_position_whole_track(self.tracks, i, channel)
            y_positions = return_track_attributes.return_puncta_y_position_whole_track(self.tracks, i, channel)

            displacements=[]

            for j in range(1,len(x_positions)):

                displacements.append(np.sqrt((x_positions[j]-x_positions[j-1])**2 + (y_positions[j]-y_positions[j-1])**2))

            self.feature_matrix.append(np.sum(displacements)/len(displacements))


    def time_to_peak_chA(self, channel):
        """Extract the time (seconds) for the intensity of channel to reach peak value; 0 if peak at first frame."""
        for i,track in enumerate(self.tracks):

            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)

            if np.argmax(chA_amplitudes) == 0:

                self.feature_matrix.append(0)

            else:

                self.feature_matrix.append(np.argmax(chA_amplitudes))

                                           
    def time_after_peak_chA(self, channel):
        """Extract the time (seconds) after the intensity of channel reaches its peak value; 0 if peak at last frame."""
        for i,track in enumerate(self.tracks):

            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)

            if np.argmax(chA_amplitudes) == (len(chA_amplitudes) - 1):

                self.feature_matrix.append(0)

            else:

                self.feature_matrix.append(len(chA_amplitudes) - 1 - np.argmax(chA_amplitudes))
          

    def time_between_peaks_chA_B(self, channels):
        """Extract the time (seconds) between the peaks of the channels."""
        for i,track in enumerate(self.tracks):
            
            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channels[0])                            
            chB_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channels[1])

            self.feature_matrix.append(np.argmax(chA_amplitudes) - np.argmax(chB_amplitudes))

                        
    def avg_int_change_to_peak_chA(self, channel):
        """Extract the average change in the first channel's intensity (a.u. fluorescence) to the peak; 0 if peak is at first frame."""
        for i,track in enumerate(self.tracks):

            change = 0
            num_changes = 0                               
            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)    

            if np.argmax(chA_amplitudes) != 0:

                for frame in range(np.argmax(chA_amplitudes)):

                    change += (chA_amplitudes[frame+1] - chA_amplitudes[frame])
                    num_changes+=1
                self.feature_matrix.append(change/num_changes)

            else:

                self.feature_matrix.append(0)

            
    def avg_int_change_after_peak_chA(self, channel):
        """Extract the average change in the channel's intensity (a.u. fluorescence) after the peak; 0 if peak is at last frame."""
        for i,track in enumerate(self.tracks):

            change = 0
            num_changes = 0                               
            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)    

            if np.argmax(chA_amplitudes) != (len(chA_amplitudes) - 1):

                for frame in range(np.argmax(chA_amplitudes),len(chA_amplitudes)-1):

                    change += (chA_amplitudes[frame+1] - chA_amplitudes[frame])
                    num_changes+=1
                self.feature_matrix.append(change/num_changes)

            else:

                self.feature_matrix.append(0)
            
    def peak_int_diff_chA_B(self, channels):
        """Extract the difference between the peak intensities (a.u. fluorescence) of the channels."""
        for i,track in enumerate(self.tracks):

            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channels[0])  
            chB_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channels[1])  

            self.feature_matrix.append(chA_amplitudes[np.argmax(chA_amplitudes)] - chB_amplitudes[np.argmax(chB_amplitudes)])
            
    def ratio_max_int_chA_B(self, channels):
        """Extract the ratio between the peak intensities (a.u. fluorescence) of the channels"""
        for i,track in enumerate(self.tracks):

            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channels[0])  
            chB_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channels[1])  

            self.feature_matrix.append(chA_amplitudes[np.argmax(chA_amplitudes)] / chB_amplitudes[np.argmax(chB_amplitudes)])

    def mean_chA(self, channel):
        """Extract the mean intensity (a.u. fluorescence) of the channel."""
        for i,track in enumerate(self.tracks):
                            
            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)  

            self.feature_matrix.append(np.mean(chA_amplitudes))
                                    
    def variation_chA(self):
        """Extract the variation in intensity (a.u. fluorescence^2) of the channel."""
        for i,track in enumerate(self.tracks):
                            
            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)  

            self.feature_matrix.append(stats.variation(chA_amplitudes))

    def skewness_chA(self, channel):
        """Extract the skewness in intensity (unitless) of the channel."""                                  
        for i,track in enumerate(self.tracks):
                            
            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)  

            self.feature_matrix.append(stats.skew(chA_amplitudes))

    def kurtosis_chA(self, channel):
        """Extract the kurtosis in intensity (unitless) of the channel."""
        for i,track in enumerate(self.tracks):

            chA_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, channel)  

            self.feature_matrix.append(stats.kurtosis(chA_amplitudes))

    def number_significant_chA(self, channel):
        """Extract the number (counts) of significant detections below the designated p-value in the channel."""
        for i,track in enumerate(self.tracks):
            

            pvals = return_track_attributes.return_pvals_detection(self.tracks,
                                           i,
                                           channel)

            self.feature_matrix.append(len(np.where(pvals<self.significant_pval_cutoff)[0]))
    
    def max_consecutive_significant_chA(self):
        """Extract the maximum number (counts) of consecutive significant detections below the designated p-value in the second channel."""        
        for i,track in enumerate(self.tracks):

            pvals = return_track_attributes.return_pvals_detection(self.tracks,
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
            
    def fraction_significant_chA(self):
        """Extract the fraction (unitless) of the second channel's lifetime that contains significant detections below the designated p-value."""
        for i,track in enumerate(self.tracks):

            pvals = return_track_attributes.return_pvals_detection(self.tracks,
                                       i,
                                       1)

            self.feature_matrix.append((pvals < self.significant_pval_cutoff).sum()/len(pvals))

                
    def fraction_peak_chA(self):
        """Extract the fraction of the whole event for the intensity of first channel to reach peak value."""
        for i,track in enumerate(self.tracks):

            ch0_amplitudes = return_track_attributes.return_track_amplitude_one_channel(self.tracks, i, 0)

            self.feature_matrix.append(np.argmax(ch0_amplitudes)/len(ch0_amplitudes))
 