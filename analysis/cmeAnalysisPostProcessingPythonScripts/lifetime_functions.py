# Cyna Shirazinejad 6/18/20
# functions to view and manipulate track lifetimes

import os
import sys
sys.path.append(os.getcwd())
from generate_index_dictionary import return_index_dictionary
from return_track_attributes import *
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
import ipywidgets as widgets
import warnings
import matplotlib
import itk
import matplotlib.animation as animation
from IPython.display import HTML
import os
from pathlib import Path
import imageio
index_dictionary = return_index_dictionary()
from IPython.display import display
from sklearn.neighbors import KDTree
from collections import Counter
from scipy.stats import mode
import scipy.stats as stats
import matplotlib.gridspec as gridspec

from generate_index_dictionary import return_index_dictionary
from return_track_attributes import (return_track_x_position, 
                                     return_designated_channels_amplitudes, 
                                     return_track_lifetime,
                                     return_track_amplitude_one_channel)
from alignment import (find_alignment_frame_protocol_max, 
                       find_alignment_frame_protocol_before_max, 
                       find_alignment_frame_protocol_after_max)
import matplotlib.pyplot as plt
import numpy as np

index_dictionary = return_index_dictionary()

def view_raw_lifetimes(tracks,
                       lifetime_limits = [0,np.Inf],
                       bins = 'default',
                       dpi=300,
                       figsize = (6,4)):
                       
    """
    Display lifetimes extracted from cmeAnalysis' ProcessedTracks.mat.
    
    Args:
        tracks (dictionary (with track indices as keys) or ndarray): tracks outputted by ProcessedTracks.mat
        lifetime_limits (list, optional): elements are floats for upper and lower bounds of lifetimes to keep
        bins (string, optional): binning option for histograms in matplotlib
        dpi (int, optional): dpi of matplotlib plot
        figsize (tuple, optional): figure size of matplotlib plot
        
    Returns:
        None
    """
    lifetimes = []
    
    for i in range(len(tracks)):
        
        if return_track_lifetime(tracks,i) > lifetime_limits[0] and return_track_lifetime(tracks,i) < lifetime_limits[1]:
            
            lifetimes.append(return_track_lifetime(tracks,i))
    
    
    
    
    print(f'The total number of events in the tracks object are: {len(tracks)}')
    print(f'The number of events plotted within defined lifetime bounds are: {len(lifetimes)}')       
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')        
    plt.title('histogram of raw lifetimes (s)')
    
    if bins == 'default':
        plt.hist(lifetimes)
    else:
        plt.hist(lifetimes, bins = bins)
    
    plt.hist(lifetimes)
    plt.xlabel('lifetime (s)')
    plt.ylabel('counts')
    plt.show()

def modify_lifetimes(tracks,
                     channel_colors=['g'],
                     interact_through_tracks=True,
                     display_channels=[0],
                     first_channel=0,
                     first_channel_location = -1,
                     first_channel_percentage_of_max=10,
                     first_moving_average=True,
                     first_average_display=True,
                     first_moving_average_window=3,
                     second_channel=1,
                     second_channel_location=0,
                     second_channel_percentage_of_max=10,
                     second_moving_average=True,
                     second_average_display=True,
                     second_moving_average_window=3,
                     print_old_lifetimes=True,
                     print_new_lifetimes=True,
                     display_final_histogram = True,
                     display_individual_modifications = True,
                     histogram_bins = 'default',
                     histogram_fig_size=(6,4),
                     histogram_dpi=300,
                     individual_fig_size=(6,4),
                     individual_dpi=100):
     
    """
    Change the 'lifetime' of an event by measuring the time between two distinct points in an intensity over time profile.
    
    Args:
        tracks (dictionary (with track indices as keys) or ndarray): tracks outputted by ProcessedTracks.mat
        channel_colors (list, optional): elements are strings corresponding to matplotlib colors for each channel
        interact_through_tracks (boolean, optional): if True, use ipywidgets to interact through each modified event one by one; if False, print out all plots
        display_channels (list, optional): elements are indices of track's channels to show on plot
        first_channel (int, optional): channel in tracks to find first reference frame of
        first_channel_location (int, optional): 0 corresponds to peak alignment, -1 corresponds to before peak alignment, 1 corresponds to after peak alignment
        first_channel_percentage_of_max (float, optional): percentage of the maximum signal in a track to align to either before or after the peak
        first channel_moving_average (boolean, optional): if True, apply moving average on intensity of first selected channel
        first_average_display (boolean, optional): if True, plot moving average
        first_moving_average_window (int, optional): number of frames for moving average
        second_channel (int, optional): channel in tracks to find second reference frame of
        second_channel_location (int, optional): 0 corresponds to peak alignment, -1 corresponds to before peak alignment, 1 corresponds to after peak alignment
        second_channel_percentage_of_max (float, optional): percentage of the maximum signal in a track to align to either before or after the peak
        second channel_moving_average (boolean, optional): if True, apply moving average on intensity of second selected channel
        second_average_display (boolean, optional): if True, plot moving average
        second_moving_average_window (int, optional): number of frames for moving average    
        print_old_lifetimes (boolean, optional): print out all old lifetimes
        print_new_lifetimes (boolean, optional): print out all new lifetime measurements
        display_final_histogram (boolean, optional): display a histogram of the the new lifetime measurements
        histogram_bins (string, optional): histogram binning option for matplotlib 
        histogram_fig_size (tuple, optional): figure size of matplotlib histogram
        histogram_dpi (int, optional): dpi of matplotlib histogram
        individual_fig_size (tuple, optional): figure size of matplotlib plots of modified lifetimes
        individual_dpi (int, optional): dpi of matplotlib plots of modified lifetimes
        
    """
    number_of_channels = len(return_track_x_position(tracks,0)) # number of channels in tracks
    
    all_channel_amplitudes = return_designated_channels_amplitudes(tracks, [*range(number_of_channels)], False) # get raw intensity data                    
    
    first_channel_amplitudes = return_designated_channels_amplitudes(tracks, [first_channel], False)  # select relevant channels
    
    second_channel_amplitudes = return_designated_channels_amplitudes(tracks, [second_channel], False)
    
    # calculate the two frames of interest for each track
    first_channel_pertinent_frames, first_channel_thresholds = retrieve_frames_and_thresholds_from_conditions(all_channel_amplitudes,
                                                                                                              first_channel_percentage_of_max,
                                                                                                              first_channel,
                                                                                                              first_channel_location,
                                                                                                              first_channel_percentage_of_max,
                                                                                                              first_moving_average,
                                                                                                              first_moving_average_window)

    second_channel_pertinent_frames, second_channel_thresholds = retrieve_frames_and_thresholds_from_conditions(all_channel_amplitudes,
                                                                                                               second_channel_percentage_of_max,
                                                                                                               second_channel,
                                                                                                               second_channel_location,
                                                                                                               second_channel_percentage_of_max, 
                                                                                                               second_moving_average,
                                                                                                               second_moving_average_window)

    # calculate the number of seconds between each frame
    interval = tracks[0][index_dictionary['index_time_frames']][0][1] - tracks[0][index_dictionary['index_time_frames']][0][0]                                               
    
    # use the frame differences and imaging interval to calculate the time interval between the two reference frames
    new_lifetimes = calculate_new_lifetimes(first_channel_pertinent_frames,
                           second_channel_pertinent_frames,
                           interval)
    
    # plots results
    if display_final_histogram:
        
        display_histogram(histogram_fig_size, 
                          histogram_dpi,
                          new_lifetimes,
                          histogram_bins)
    
    if display_individual_modifications:
        number_of_channels = len(return_track_x_position(tracks,0))
        if interact_through_tracks:

            interact(display_individual_lifetimes,
                     tracks=fixed(tracks),
                     track_ID = widgets.SelectionSlider(description='Track ID:',   options=list(range(len(tracks))), value = 0),
                     new_lifetimes=fixed(new_lifetimes),
                     channel_colors=fixed(channel_colors),
                     display_channels=fixed(display_channels),
                     all_channel_amplitudes=fixed(all_channel_amplitudes),
                     first_channel=fixed(first_channel),
                     second_channel=fixed(second_channel),
                     first_channel_amplitudes=fixed(first_channel_amplitudes),
                     second_channel_amplitudes=fixed(second_channel_amplitudes),
                     first_channel_pertinent_frames=fixed(first_channel_pertinent_frames),
                     second_channel_pertinent_frames=fixed(second_channel_pertinent_frames),
                     first_channel_thresholds=fixed(first_channel_thresholds),
                     second_channel_thresholds=fixed(second_channel_thresholds),
                     first_moving_average=fixed(first_moving_average),
                     first_moving_average_display=fixed(first_average_display),
                     first_moving_average_window=fixed(first_moving_average_window),
                     second_moving_average=fixed(second_moving_average),
                     second_moving_average_display=fixed(second_average_display),
                     second_moving_average_window=fixed(second_moving_average_window),
                     print_old_lifetimes=fixed(print_old_lifetimes),
                     print_new_lifetimes=fixed(print_new_lifetimes),
                     individual_fig_size=fixed(individual_fig_size),
                     individual_dpi=fixed(individual_dpi))
        else:
            
            for track_ID in range(len(tracks)):

                display_individual_lifetimes(tracks,
                                     track_ID,
                                     new_lifetimes,
                                     channel_colors,
                                     display_channels,
                                     all_channel_amplitudes,
                                     first_channel,
                                     second_channel,
                                     first_channel_amplitudes,
                                     second_channel_amplitudes,
                                     first_channel_pertinent_frames,
                                     second_channel_pertinent_frames,
                                     first_channel_thresholds,
                                     second_channel_thresholds,
                                     first_moving_average,
                                     first_moving_average_display,
                                     first_moving_average_window,
                                     second_moving_average,
                                     second_moving_average_display,
                                     second_moving_average_window,
                                     print_old_lifetimes,
                                     print_new_lifetimes,
                                     individual_fig_size,
                                     individual_dpi)

    print('The new lifetimes are:')       
    return new_lifetimes

def display_individual_lifetimes(tracks,
                                 track_ID,
                                 new_lifetimes,
                                 channel_colors,
                                 display_channels,
                                 all_channel_amplitudes,
                                 first_channel,
                                 second_channel,
                                 first_channel_amplitudes,
                                 second_channel_amplitudes,
                                 first_channel_pertinent_frames,
                                 second_channel_pertinent_frames,
                                 first_channel_thresholds,
                                 second_channel_thresholds,
                                 first_moving_average,
                                 first_moving_average_display,
                                 first_moving_average_window,
                                 second_moving_average,
                                 second_moving_average_display,
                                 second_moving_average_window,
                                 print_old_lifetimes,
                                 print_new_lifetimes,
                                 individual_fig_size,
                                 individual_dpi):
    
    """
    Plot the modified lifetimes.
    
    Args:
        tracks (dictionary (with track indices as keys) or ndarray): tracks outputted by ProcessedTracks.mat
        track_ID (int): index of track whose lifetime is being plotted
        new_lifetimes (list): elements are floats corresponding to the modified lifetime
        channel_colors (list): elements are strings corresponding to matplotlib colors for each channel
        display_channels (list): elements are indices of track's channels to show on plot
        all_channel_amplitudes (list): elements correspond to individual tracks with lists for each channel available for the track
        first_channel (int): channel in tracks to find first reference frame of
        first_channel_location (int): 0 corresponds to peak alignment, -1 corresponds to before peak alignment, 1 corresponds to after peak alignment
        first_channel_percentage_of_max (float): percentage of the maximum signal in a track to align to either before or after the peak
        first channel_moving_average (boolean): if True, apply moving average on intensity of first selected channel
        first_average_display (boolean): if True, plot moving average
        first_moving_average_window (int): number of frames for moving average
        second_channel (int): channel in tracks to find second reference frame of
        second_channel_location (int): 0 corresponds to peak alignment, -1 corresponds to before peak alignment, 1 corresponds to after peak alignment
        second_channel_percentage_of_max (float): percentage of the maximum signal in a track to align to either before or after the peak
        second channel_moving_average (boolean): if True, apply moving average on intensity of second selected channel
        second_average_display (boolean): if True, plot moving average
        second_moving_average_window (int): number of frames for moving average    
        print_old_lifetimes (boolean): print out all old lifetimes
        print_new_lifetimes (boolean): print out all new lifetime measurements
        display_final_histogram (boolean): display a histogram of the the new lifetime measurements
        histogram_bins (string): histogram binning option for matplotlib 
        histogram_fig_size (tuple): figure size of matplotlib histogram
        histogram_dpi (int): dpi of matplotlib histogram
        individual_fig_size (tuple): figure size of matplotlib plots of modified lifetimes
        individual_dpi (int): dpi of matplotlib plots of modified lifetimes
        
    """
    i=track_ID
    number_of_channels = len(return_track_x_position(tracks,0))


    plt.figure(num=None, figsize=individual_fig_size, dpi=individual_dpi, facecolor='w', edgecolor='k')
    print()
    print(f'The trackID of the following track is {i}')
    legend_label = []

#         track_lifetime = return_track_lifetime(tracks,i)

    if print_old_lifetimes:

        print(f'The old lifetime of the track is: {len(return_track_amplitude_one_channel(tracks,i,0))} s')

    if print_new_lifetimes:

        print(f'The new lifetime of the track is: {new_lifetimes[i]} s')

    for j in range(number_of_channels):

        if j in display_channels:
#                 print(len(all_channel_amplitudes[i][j]))
            plt.plot(all_channel_amplitudes[i][j], color=channel_colors[j])

            legend_label.append('ch' + str(j))

    if first_moving_average_display:
#             print(first_channel)

        plt.plot(moving_average(all_channel_amplitudes[i][first_channel],first_moving_average_window),linestyle=(0, (3, 1, 1, 1, 1, 1)))
        legend_label.append('first selected channel smoothed')

    if second_moving_average_display:
#             print('test')
        plt.plot(moving_average(all_channel_amplitudes[i][second_channel],second_moving_average_window),linestyle=(0, (3, 1, 1, 1, 1, 1)))
        legend_label.append('second selected channel smoothed')

    plt.axhline(y=first_channel_thresholds[i], color='k', linestyle=':')
    legend_label.append('first threshold')

    plt.axhline(y=second_channel_thresholds[i], color='tab:brown', linestyle='-.')
    legend_label.append('second_threshold')    

    plt.axvline(x=first_channel_pertinent_frames[i],color='k', linestyle=':')
    legend_label.append('first selection')
    plt.axvline(x=second_channel_pertinent_frames[i],color='tab:brown', linestyle='-.')
    legend_label.append('second selection')

    plt.plot(first_channel_pertinent_frames[i],first_channel_thresholds[i],marker='*',markerSize=20,color='r')
    plt.plot(second_channel_pertinent_frames[i],second_channel_thresholds[i],marker='*',markerSize=20,color='g')
    legend_label.append('first channel intersection')
    legend_label.append('second channel intersection')
    plt.legend(legend_label)
    plt.title('modified lifetime')
    plt.xlabel('time (s)')
    plt.ylabel('au fluorescence intensity')
    plt.show()
    
def display_histogram(histogram_fig_size, 
                      histogram_dpi,
                      new_lifetimes,
                      histogram_bins):
    
    
    """
    Plot histogram of provided values.
    
    Args:
        histogram_fig_size (tuple): figure size of matplotlib plot
        histogram_dpi (int): dpi of matplotlib plot
        new_lifetimes (list): elements are floats corresponding to values to plot in the histogram
        histogram_bins (string): histogram binning option for matplotlib
        
    Returns:
        None
    """   
    plt.figure(num=None, figsize=histogram_fig_size, dpi=histogram_dpi, facecolor='w', edgecolor='k')
    if histogram_bins=='default':
        plt.hist(new_lifetimes)
    else:
        plt.hist(new_lifetimes,bins=histogram_bins)
    plt.xlabel('lifetime (s)')
    plt.ylabel('frequency')
    plt.title('modified lifetime histogram')
    plt.show()
        
def calculate_new_lifetimes(first_frame_locations,
                           second_frame_locations,
                           interval):
    """Calculate the time (in seconds) between the two given frames."""
    return interval * np.abs((np.asarray(second_frame_locations) - np.asarray(first_frame_locations)))

    
            

        
        
        
def retrieve_frames_and_thresholds_from_conditions(amplitudes,
                                                   alignment_percentage,
                                                   channel,
                                                   channel_location,
                                                   channel_percentage_of_max,
                                                   moving_average_convolution,
                                                   window_size):
                                    
    
    
    """Calculate the two frames of reference in the track based on the position instructions provided."""
    frames_return = []
    channel_thresholds = []
    
    for i in range(len(amplitudes)):
        
        current_amplitudes = amplitudes[i][channel]

        if moving_average_convolution:

            current_amplitudes = moving_average(np.ndarray.flatten(np.asarray(current_amplitudes)),window_size)

        if channel_location==0:
            
            frames_return.append(find_alignment_frame_protocol_max(current_amplitudes)[0])
        
        elif channel_location==-1:
            
            frames_return.append(find_alignment_frame_protocol_before_max(current_amplitudes,channel_percentage_of_max)[0])
            
        elif channel_location==1:
            
            frames_return.append(find_alignment_frame_protocol_after_max(current_amplitudes,channel_percentage_of_max)[0])
            
    return frames_return, channel_thresholds


def moving_average(x, w):
    """Apply a moving average on a signal 'x' with a window size 'w.'"""
    import numpy as np
    return np.convolve(x, np.ones(w), 'valid')/w