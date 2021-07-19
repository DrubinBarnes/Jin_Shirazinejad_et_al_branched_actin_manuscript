# Cyna Shirazinejad, last modified 10/8/20
# alignment functions for tracking data of arbitrary number of channels
#
import os
import sys
sys.path.append(os.getcwd())
from return_track_attributes import return_designated_channels_amplitudes
import numpy as np
import matplotlib.pyplot as plt
from return_track_attributes import return_track_x_position


def align_tracks(tracks, 
                 channel_colors, 
                 selected_ids = 'all',
                 alignment_channel = 0,
                 alignment_protocol = 0, 
                 alignment_percentage = 100,
                 display_channels=[0],
                 channels_to_average=[0],
                 scale_to_one_before_alignment=False, 
                 scale_to_one_all=False,
                 padding=False, 
                 padding_amount=0, 
                 pad_with_zero = False,
                 stds = 0.25,
                 ylimits=[-0.1, 1.1],
                 fig_size = (3,2),
                 dpi = 300):

    """
    Align track to a designated feature in a channel.
    
    Args:
        tracks (dictionary (with track indices as keys) or ndarray): tracks outputted by ProcessedTracks.mat
        channel_colors (list, optional): elements are strings corresponding to matplotlib colors for plotting channels
        selected_ids (list): elements are ints corresponding to the indices of tracks to align and plot
        alignment_channel (int, optional): channel of tracks to align features to
        alignment_protocol (int, optional): 0 corresponds to peak alignment, -1 corresponds to before peak alignment, 1 corresponds to after peak alignment
        alignment_percentage(float, optional): percentage of the maximum signal in a track to align to either before or after the peak
        display_channels (list, optional): indices of channels in tracks to plot
        channels_to_average (list, optional): indices of channels to include in alignment
        scale_to_one_before_alignment (boolean, optional): scale maximum intensity of all channels to one before alignment
        scale_to_one_all (boolean, optional): scale maximum intensity of all channels to one after alignment
        padding (boolean, optional): pad ends of aligned tracks with extra frames to each end
        padding_amount(int, optional): amount of frames to pad to each end of aligned tracks
        pad_with_zero (boolean, optional): pad ends of signals with nans if False, otherwise pad with zeros
        stds (float, optional): number of standard deviations around the mean at each frame of aligned tracks to plot
        ylimits (list, optional): ylimits of matplotlib plot
        fig_size (tuple, optional): size of matplotlib plot
        dpi (int, optional): dpi of matplotlib plot
        
    Returns:
        None
        
    """
        
    number_of_channels = len(return_track_x_position(tracks,0)) # the number of channels in the track set
    
    if selected_ids == 'all':
        selected_ids = list(range(len(tracks))) # extract amplitudes of all tracks in set
        channel_amplitudes = return_designated_channels_amplitudes(tracks, selected_ids, channels_to_average, scale_to_one_before_alignment)
    
    else: # extrat amplitudes of designated tracks
    
        channel_amplitudes = return_designated_channels_amplitudes(tracks, selected_ids, channels_to_average, scale_to_one_before_alignment)
    
    # find the number of frames before and after each track to pad with based on alignment protocol    
    frames_before_alignment, frames_after_alignment = find_frames_around_alignment_point(channel_amplitudes,
                                                                               alignment_channel, 
                                                                               alignment_protocol,
                                                                               alignment_percentage)

    
    max_frames_around_alignment = np.max([np.max(frames_before_alignment), np.max(frames_after_alignment)])
    
    # padding amount must exceed the number of minimum frames to add to beginning and end of each track
    if padding and (max_frames_around_alignment > padding_amount or max_frames_around_alignment > padding_amount):

        raise Exception('the number of padded frames to center your aligned feature is less than the amount of default padded frames. increase the padded value to be greater than or equal to ' + str(max_frames_around_alignment))
   
    if not padding:
                        
        padding_amount = max_frames_around_alignment    
    
    # shift the amplitudes by padding the ends, resulting in aligned intensities
    shifted_amplitudes = return_shifted_amplitudes(channel_amplitudes, 
                                                   frames_before_alignment, 
                                                   frames_after_alignment, 
                                                   padding_amount, 
                                                   pad_with_zero)


    shifted_amplitudes_averaged, shifted_amplitudes_std = return_averaged_amplitudes(shifted_amplitudes) # find mean and standard deviation of aligned intensities

    if scale_to_one_all:
        # scale final aligned intensities to one in each channel
        shifted_amplitudes_averaged, shifted_amplitudes_std = scale_all_to_one(shifted_amplitudes_averaged, shifted_amplitudes_std)

    plot_aligned_features(shifted_amplitudes_averaged, 
                          shifted_amplitudes_std, 
                          padding_amount, 
                          channel_colors,
                          display_channels,
                          stds,
                          ylimits,
                          dpi,
                          fig_size)

def scale_all_to_one(mean_amplitudes, std_amplitudes):
    """Scale each track's amplitude and standard deviation to one by the maximum of each track's amplitude."""
    scaled_averaged_amplitudes = []
    scaled_std_amplitudes = []
    
    for i in range(len(mean_amplitudes)):
        
        max_signal = np.max(mean_amplitudes[i])
        
        scaled_averaged_amplitudes.append(1/max_signal * mean_amplitudes[i])
        scaled_std_amplitudes.append(1/max_signal * std_amplitudes[i])
        
    return scaled_averaged_amplitudes, scaled_std_amplitudes
        

    
def return_shifted_amplitudes(amplitudes, 
                              frames_before_alignment, 
                              frames_after_alignment, 
                              padding_amount,
                              pad_with_zero):
    """Shift amplitudes to alignment point by padding ends of intensities with zero or nans."""
    shifted_amplitudes = []

    for i in range(len(amplitudes)):

        temp_track_amplitudes = []

        for j in range(len(amplitudes[i])):


            temp_track_amplitudes.append(pad_amplitudes(amplitudes[i][j],
                                                        frames_before_alignment[i],
                                                        frames_after_alignment[i],
                                                        padding_amount,pad_with_zero))
        
        shifted_amplitudes.append(np.asarray(temp_track_amplitudes))
        
    return shifted_amplitudes
    
                                                   
def return_averaged_amplitudes(amplitudes):
    """Return the average amplitude for a channel along with its standard deviation."""
    amplitudes = np.asarray(amplitudes)
    
    averaged_amplitudes = []
    
    amplitudes_averaged = np.nan_to_num(np.nanmean(amplitudes,axis=0,dtype=np.float64))
    amplitudes_std = np.nan_to_num(np.nanstd(amplitudes,axis=0,dtype=np.float64))
    
    return amplitudes_averaged, amplitudes_std
    
    
def plot_aligned_features(shifted_amplitudes_averaged, 
                          shifted_amplitudes_std, 
                          padding_amount, 
                          channel_colors,
                          display_channels,
                          stds,
                          ylimits,
                          dpi,
                          fig_size,
                          xlimits=[-100,100]):
    """Create a matplotlib plot of the aligned intensities."""
    plt.figure(num=None, figsize=fig_size, dpi=dpi, facecolor='w', edgecolor='k')
    legend_label = []
    for i in range(len(shifted_amplitudes_averaged)):
        
        if i in display_channels:
            
            plt.plot(range(-padding_amount,padding_amount + 1),shifted_amplitudes_averaged[i],channel_colors[i])
            plt.fill_between(range(-padding_amount,padding_amount + 1),
                             shifted_amplitudes_averaged[i] - stds * shifted_amplitudes_std[i],
                             shifted_amplitudes_averaged[i] + stds * shifted_amplitudes_std[i],
                             color = channel_colors[i],
                             alpha = 0.2)
            legend_label.append('ch' + str(i))
    plt.legend(legend_label)        
    plt.ylim(ylimits) 
    plt.xlim(xlimits)
    plt.show()
    
def pad_amplitudes(amplitudes, 
                   frames_before, 
                   frames_after, 
                   padding_amount,
                   pad_with_zero):
    """Add padded frames to amplitude array."""
    frames_before = padding_amount - frames_before
    frames_after = padding_amount - frames_after
    
    if pad_with_zero:
        pad_val = 0
    else:
        pad_val = np.nan
    vector_before = np.full(frames_before, pad_val)
    vector_after = np.full(frames_after, pad_val)
    
    return np.concatenate((vector_before, amplitudes, vector_after), axis = 0)

                
    
def find_frames_around_alignment_point(channel_amplitudes,
                                       alignment_channel, 
                                       alignment_protocol,
                                       alignment_percentage):
    """Return the number of frames before and after the designated alignment protocol point."""
    frames_before_alignment = []
    frames_after_alignment = []
    
    for i in range(len(channel_amplitudes)):
        
        if alignment_protocol == 0:
            
            frames_temp = find_alignment_frame_protocol_max(channel_amplitudes[i][alignment_channel])

            frames_before_alignment.append(frames_temp[0])
            frames_after_alignment.append(frames_temp[1])
            
        elif alignment_protocol == -1:
            
            frames_temp = find_alignment_frame_protocol_before_max(channel_amplitudes[i][alignment_channel],
                                                                   alignment_percentage)
            
            frames_before_alignment.append(frames_temp[0])
            frames_after_alignment.append(frames_temp[1])

        elif alignment_protocol == 1:
            
            frames_temp = find_alignment_frame_protocol_after_max(channel_amplitudes[i][alignment_channel],
                                                                   alignment_percentage)
            
            frames_before_alignment.append(frames_temp[0])
            frames_after_alignment.append(frames_temp[1])
            
    return frames_before_alignment, frames_after_alignment


def find_alignment_frame_protocol_max(amplitudes):
    """Return the number of frames before peak and after peak."""
    return (np.argmax(amplitudes), len(amplitudes) - np.argmax(amplitudes) - 1)


def find_alignment_frame_protocol_before_max(amplitudes, 
                                             alignment_percentage):
    """Return the number of frames before and after the designated percentage of maximum signal before the peak."""
#     print(np.argmax(amplitudes))
#     print(len(amplitudes))
    value_fraction = amplitudes[np.argmax(amplitudes)]*alignment_percentage/100
    
    if np.argmax(amplitudes) == 0:
        
        return (0, len(amplitudes) - 1)
    
    else:
        
        idx = (np.abs(amplitudes[0:np.argmax(amplitudes)] - value_fraction)).argmin()
        return (idx,len(amplitudes) - idx - 1)

    
def find_alignment_frame_protocol_after_max(amplitudes, 
                                            alignment_percentage):
    """Return the number of frames before and after the designated percentage of maximum signal after the peak."""
    value_fraction = amplitudes[np.argmax(amplitudes)]*alignment_percentage/100
    
    if np.argmax(amplitudes) == (len(amplitudes) - 1):
        
        return (len(amplitudes) - 1, 0)
    
    else:
        
        idx = (np.abs(amplitudes[np.argmax(amplitudes):] - value_fraction)).argmin()
        return (np.argmax(amplitudes) + idx, len(amplitudes) - idx - np.argmax(amplitudes) - 1)